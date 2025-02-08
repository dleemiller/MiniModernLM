import logging
import os
from dataclasses import asdict

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset

from minilmv2.config import DataArguments, ModelArguments
from minilmv2.minilmv2 import MiniLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_model_info(model, name):
    """Log model architecture and parameter count."""
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{name} total parameters: {num_params:,}")
    logger.info(f"{name} trainable parameters: {trainable_params:,}")
    logger.info(f"{name} architecture:\n{model}")


def verify_training_args(training_args):
    """Ensure all necessary training arguments are set."""
    if not training_args.logging_dir:
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")

    # Set default logging configuration if not specified
    if not hasattr(training_args, "logging_strategy"):
        training_args.logging_strategy = "steps"
    if not hasattr(training_args, "logging_first_step"):
        training_args.logging_first_step = True

    # Enable gradient logging
    training_args.logging_nan_inf_filter = True
    training_args.logging_gradient_norm = True

    return training_args


def main():
    # Parse dataclass arguments
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Verify and setup training arguments
    training_args = verify_training_args(training_args)

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Log all arguments
    logger.info(f"Data args: {asdict(data_args)}")
    logger.info(f"Model args: {asdict(model_args)}")
    logger.info(f"Training args: {asdict(training_args)}")

    # --- Teacher Model (and Config) ---
    logger.info("Loading teacher model...")
    teacher = AutoModel.from_pretrained(model_args.teacher_model_id)
    teacher_config = AutoConfig.from_pretrained(model_args.teacher_model_id)
    log_model_info(teacher, "Teacher")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.teacher_model_id, use_fast=True
    )

    # --- Student Model Configuration ---
    logger.info("Configuring student model...")
    student_config = AutoConfig.from_pretrained(model_args.teacher_model_id)
    student_config.hidden_size = model_args.student_hidden_size
    student_config.intermediate_size = model_args.student_intermediate_size
    student_config.mlp_dropout = model_args.student_mlp_dropout
    student_config.num_hidden_layers = model_args.student_num_layers
    student_config.num_attention_heads = model_args.student_attention_heads

    if hasattr(student_config, "global_attn_every_n_layers"):
        student_config.global_attn_every_n_layers = (
            model_args.student_global_attn_every_n_layers
        )

    # Explicitly set the L parameter in the student config
    student_config.L = model_args.L

    # Create the student model from the modified configuration
    student = AutoModel.from_config(student_config)
    log_model_info(student, "Student")

    # --- Distillation Model ---
    logger.info("Initializing MiniLMv2...")
    distiller = MiniLM(
        teacher=teacher,
        student=student,
        L=model_args.L,  # Teacher layer to distill from
        M=student_config.num_hidden_layers,  # Student layer count
        relations=model_args.relations,  # Directly use dictionary
        A_r=model_args.num_relation_heads,
    )

    # --- Dataset Loading without Pre-Tokenization ---
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    training_args.remove_unused_columns = False
    raw_train_dataset = load_dataset(
        data_args.dataset_name,
        split=data_args.train_split,
        cache_dir=data_args.cache_dir,
    )
    raw_val_dataset = load_dataset(
        data_args.dataset_name, split=data_args.val_split, cache_dir=data_args.cache_dir
    )

    logger.info(f"Train dataset size: {len(raw_train_dataset)}")
    logger.info(f"Validation dataset size: {len(raw_val_dataset)}")

    # --- Dynamic Tokenization in the Data Collator ---
    def dynamic_tokenization_collate_fn(examples):
        texts = [example[data_args.text_column] for example in examples]
        return tokenizer(
            texts,
            truncation=True,
            max_length=data_args.max_seq_len,
            padding="longest",
            return_tensors="pt",
        )

    # --- Initialize the Trainer ---
    trainer = Trainer(
        model=distiller,
        args=training_args,
        train_dataset=raw_train_dataset,
        eval_dataset=raw_val_dataset,
        data_collator=dynamic_tokenization_collate_fn,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],  # Reverted to standard TensorBoard callback
    )

    # --- Training ---
    logger.info("Starting training")
    train_result = trainer.train()

    # Log final training statistics
    logger.info("Training completed. Final stats:")
    for key, value in train_result.metrics.items():
        logger.info(f"{key}: {value}")

    # Save the final model
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save training state
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )


if __name__ == "__main__":
    main()
