# config.py
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="skymizer/fineweb-edu-dedup-45B",
        metadata={"help": "The Hugging Face dataset identifier."},
    )
    cache_dir: str = field(default="/home/datasets")
    train_split: str = field(
        default="train", metadata={"help": "Name of the training split."}
    )
    val_split: str = field(
        default="test", metadata={"help": "Name of the validation split."}
    )
    max_seq_len: int = field(
        default=256, metadata={"help": "Maximum sequence length for tokenization."}
    )
    text_column: str = field(
        default="text", metadata={"help": "Column containing the raw text."}
    )


@dataclass
class ModelArguments:
    teacher_model_id: str = field(
        default="answerdotai/ModernBERT-large",
        metadata={"help": "The teacher model identifier on Hugging Face."},
    )
    student_hidden_size: int = field(
        default=384, metadata={"help": "Hidden size for the student model."}
    )
    student_intermediate_size: int = field(
        default=640, metadata={"help": "Intermediate size for the student model."}
    )
    student_num_layers: int = field(
        default=5,
        metadata={"help": "Number of transformer layers for the student model."},
    )
    student_attention_heads: int = field(
        default=8, metadata={"help": "Number of attention heads for the student model."}
    )
    student_global_attn_every_n_layers: int = field(
        default=2, metadata={"help": "Global attention frequency for the student."}
    )
    student_mlp_dropout: int = field(
        default=0.1, metadata={"help": "MLP dropout rate."}
    )
    L: int = field(
        default=22, metadata={"help": "Teacher layer to distill (1-based index)."}
    )
    relations: Dict[Tuple[int, int], float] = field(
        default_factory=lambda: {(1, 1): 1 / 3, (2, 2): 1 / 3, (3, 3): 1 / 3},
        metadata={"help": "Relation pairs and weights for MiniLMv2."},
    )
    num_relation_heads: int = field(
        default=4,
        metadata={"help": "Number of relation heads for MiniLMv2 distillation."},
    )
