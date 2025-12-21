	export MARITALK_API_SECRET_KEY='106342464522172013028_53c19f0449d9d8d6'
	
	experiment_label="poeta_v2_full_qwen32b_2"
	model_config="configs/vllm_qwen32b.json"
	task_config="configs/poeta_v2_full_2.json"

	python scripts/bulk_evaluation.py \
	  --model_config "$model_config" \
	  --task_configs "$task_config" \
	  --experiment_name "$experiment_label"
