#!/bin/bash
CMD="python -m train"

SEED=42
OUTPUT_DIR="./out"
LOG_DIR="${OUTPUT_DIR}/logs"

ARGS="--per_device_train_batch_size 256 \
      --learning_rate 2e-4 \
      --adam_epsilon 1e-6 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --weight_decay 0.01 \
      --save_steps 5000 \
      --logging_steps 10 \
      --warmup_steps 4000 \
      --output_dir ${OUTPUT_DIR} \
      --logging_dir ${LOG_DIR} \
      --logging_strategy steps \
      --logging_first_step true \
      --bf16 \
      --lr_scheduler_type linear \
      --seed ${SEED} \
"

# Create logging directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Run the command
$CMD $ARGS

# Print TensorBoard startup command for convenience
echo -e "\nTo view training logs, run:"
echo "tensorboard --logdir ${LOG_DIR}"
