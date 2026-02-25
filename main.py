import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--prompt_modes', default="dynamic-random")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    parser.add_argument('--description_dict_path', default=None)
    parser.add_argument('--conversation_template', type=str, default=None)
    parser.add_argument('--prompt_as_single_user_message', action="store_true")
    parser.add_argument('--check_integrity', action="store_true")
    parser.add_argument('--pass_k', type=int, default=1)
    parser.add_argument('--pass_n', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    assert not args.provide_description  # not implemented
    assert args.pass_k >= 1, "--pass_k must be >= 1"
    assert args.pass_n >= 1, "--pass_n must be >= 1"
    assert args.pass_n >= args.pass_k, "--pass_n must be >= --pass_k"

    if args.pass_k > 1 and not args.no_cache:
        print("WARNING: pass@k with cache enabled can reuse stale generations across runs. Prefer --no_cache.")
    
    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
        
        
    prompt_modes = args.prompt_modes.split(",")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, 'r') as f:
            description_dict = json.load(f)

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        # if output path is a file (e.g "output.txt") in the current directory, output_dir will be empty
        if output_dir == "":
            output_dir = "."
    else:
        output_dir = None
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        prompt_modes=prompt_modes,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        conversation_template=args.conversation_template,
        prompt_as_single_user_message=args.prompt_as_single_user_message,
        check_integrity=args.check_integrity,
        output_dir=output_dir,
        pass_k=args.pass_k,
        pass_n=args.pass_n,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    dumped = json.dumps(results, indent=2)
    
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}, "
        f"pass_k: {args.pass_k}, pass_n: {args.pass_n}, temperature: {args.temperature}, top_p: {args.top_p}"
    )
    print(evaluator.make_table(results))

if __name__ == "__main__":
    main()
