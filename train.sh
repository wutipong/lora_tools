#!/bin/sh

source /home/noom/lora_tools/venv/bin/activate

/home/noom/lora_tools/venv/bin/accelerate launch --quiet \
	--config_file=/home/noom/lora_tools/accelerate_config.yaml \
	--num_cpu_threads_per_process=1 \
	/home/noom/Loras/sd-scripts/train_network_xl_wrapper.py \
	--dataset_config=/home/noom/lora_tools/workspace/$TARGET/dataset_config.toml \
	--config_file=/home/noom/lora_tools/workspace/$TARGET/training_config.toml
