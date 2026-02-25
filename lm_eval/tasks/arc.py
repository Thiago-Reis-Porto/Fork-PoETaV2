"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoicePromptSelectionTask
import inspect
import math
import re
import lm_eval.datasets.gsm8k.gsm8k
from pathlib import Path
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import collections
import json
import numpy as np
import os
import re
from zipfile import ZipFile
from pathlib import Path
import subprocess
from datasets import load_dataset
import textwrap
from lm_eval.base import rf, MultipleChoicePromptSelectionTask
from lm_eval.metrics import mean
import random

_CITATION = """
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# maritaaca imp
class ARC_CHALLENGE_greedy(MultipleChoicePromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "allenai/ai2_arc"
    DATASET_NAME = None
    DATASET_CONFIG = "ARC-Challenge"
    
    language = "en"
    tag = None
        
    rnd=random.Random(42)
    cot=False

    manual_examples = []
    pass_k = 1
    pass_n = 1

    def set_pass_k(self, pass_k):
        self.pass_k = max(1, int(pass_k))

    def set_pass_n(self, pass_n):
        self.pass_n = max(1, int(pass_n))

    def set_pass_params(self, pass_k, pass_n):
        self.set_pass_k(pass_k)
        self.set_pass_n(pass_n)

    @staticmethod
    def _pass_at_k_estimator(n, c, k):
        if c <= 0:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset = load_dataset(self.DATASET_PATH, self.DATASET_CONFIG)["test"]

        self.dataset = collections.defaultdict(list)
        
        fields_to_verify=[
            "question",
            "choices",
        ]
        
        

        for example in dataset:
            
            valid=True
            for field in fields_to_verify:
                if field not in example:
                    valid=False
                    
            #check if labels are letters
            if not all(label.isalpha() for label in example["choices"]["label"]):
                valid=False
                    
            if valid:
                self.dataset["test"].append(example)

    
        self.dataset["test"] = list(map(self._process_doc, self.dataset["test"]))


    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["test"]

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        
        question=doc["question"]
        alternatives_text=doc["choices"]["text"]
        
        choices_letters=doc["choices"]["label"]
        
        correct_index=choices_letters.index(doc["answerKey"])
        
        correct_letter=doc["answerKey"]
        
        formatted_alternatives = [
            f"{letter}. {text}" for letter, text in zip(choices_letters, alternatives_text)
            ]
        
        alternative_string="\n".join(formatted_alternatives)
                
        formatted_question=f"""{question}\n{alternative_string}"""

        return {
            "formatted_question": formatted_question,
            "gold_index": correct_index,
            "gold_letter": correct_letter,
            "choices": choices_letters,
            "choices_text": alternatives_text,
            "id": doc["id"],
        }

    def doc_to_text(self, doc):
        answer_handle = self.get_answer_handle(doc)
        return doc["formatted_question"] + "\n" + answer_handle
    
    def get_answer_handle(self, doc):
        if self.language == "en":
            return "Answer:"
        elif self.language == "pt":
            return "Resposta:"
        else:
            raise ValueError(f"Invalid language: {self.language}")

    def doc_to_target(self, doc):
        return " "+ doc["gold_letter"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        if self.pass_n > 1 or self.pass_k > 1:
            return [rf.greedy_until(ctx, ["\n"]) for _ in range(self.pass_n)]
        return rf.greedy_until(ctx, ["\n"])
    
    
    def best_effort_match(self, gold, pred):
        
        if pred == gold:
            return True, pred
        
        # check if gold and pred match removing spaces

        pred=pred.lower().strip()
        gold=gold.lower().strip()
        
        
        if pred == gold:
            return True, pred
        

        # try to find first ocorrence of "X." where X is A,B,C or D
        pred_letter=re.search(r"[a-d]\.",pred)
        
        if pred_letter is not None:
            # matched something like "X." where X is A,B,C or D
            pred_letter=pred_letter.group(0).replace(".","").strip()
            if pred_letter==gold:
                return True, pred
                    
        return False, pred
        

    def _score_prediction(self, doc, pred_answer):
        pred_answer = "" if pred_answer is None else str(pred_answer).strip()
        gold = doc["gold_letter"]
        match_result, best_pred = self.best_effort_match(gold, pred_answer)
        unknown = False
        debug_info = {
            "gold": gold,
            "pred": pred_answer,
            "best_effort_pred": best_pred,
        }
        return match_result, unknown, debug_info

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        preds = list(results) if isinstance(results, (list, tuple)) else [results]
        target_n = self.pass_n if (self.pass_n > 1 or self.pass_k > 1) else 1
        preds = preds[:target_n]

        scored = [self._score_prediction(doc, pred) for pred in preds]
        primary_correct, primary_unknown, primary_debug = scored[0]
        output = {
            "acc": 1.0 if primary_correct else 0.0,
            "debug_info": primary_debug,
            "unknown_pred": 1.0 if primary_unknown else 0.0,
        }

        if self.pass_n > 1 or self.pass_k > 1:
            c = sum(1 for is_correct, _, _ in scored if is_correct)
            output[f"acc_pass@{self.pass_k}"] = self._pass_at_k_estimator(
                n=target_n, c=c, k=self.pass_k
            )
            output[f"unknown_pred_pass@{self.pass_k}"] = (
                1.0 if all(is_unknown for _, is_unknown, _ in scored) else 0.0
            )
            output["c_mean"] = float(c)
            output["n"] = float(target_n)
            output["k"] = float(self.pass_k)

        return output

    def higher_is_better(self):
        metrics = {
            "acc": True,
            "unknown_pred": False,
        }
        if self.pass_n > 1 or self.pass_k > 1:
            metrics[f"acc_pass@{self.pass_k}"] = True
            metrics[f"unknown_pred_pass@{self.pass_k}"] = False
            metrics["c_mean"] = True
            metrics["n"] = False
            metrics["k"] = False
        return metrics

    def aggregation(self):
        metrics = {
            "acc": mean,
            "unknown_pred": mean,
        }
        if self.pass_n > 1 or self.pass_k > 1:
            metrics[f"acc_pass@{self.pass_k}"] = mean
            metrics[f"unknown_pred_pass@{self.pass_k}"] = mean
            metrics["c_mean"] = mean
            metrics["n"] = mean
            metrics["k"] = mean
        return metrics

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        # For each doc, limit the self._training_docs to examples from other exams.
        # We also remove the top-10 largest documents from the list of prompt candidates.
        self._training_docs = []
        for d in self.training_docs():
            if d["id"] != doc["id"]:
                self._training_docs.append(d)
        if prompt_mode == "dynamic-random":
            return rnd.sample(self._training_docs, k)

        elif prompt_mode == "fixed":
            return rnd.sample(self._training_docs[:k], k)

        elif prompt_mode == "manual":
            _manual_docs = []
            for d in self.manual_examples:
                if d["exam"] != doc["exam"]:
                    _manual_docs.append(d)
            assert k <= len(_manual_docs), (
                f"The number of manual_examples is not enough to satisfy "
                f"num_fewshot={k}. Please, include more examples."
            )
            return rnd.sample(_manual_docs, k)

        else:
            print(
                'Please set prompt_mode as "fixed", "dynamic-random"'
            )



class ARC_CHALLENGE_greedy_PT(ARC_CHALLENGE_greedy):
    language = "pt"
    DATASET_PATH = "maritaca-ai/ai2_arc_pt"
    
class ARC_EASY_greedy(ARC_CHALLENGE_greedy):
    DATASET_CONFIG = "ARC-Easy"
    
class ARC_EASY_greedy_PT(ARC_CHALLENGE_greedy_PT):
    DATASET_PATH = "maritaca-ai/ai2_arc_easy"
    DATASET_CONFIG = "ARC-Easy"
    
    
