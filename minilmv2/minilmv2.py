import logging
import math
from typing import Dict, Tuple

import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA

# Import the combined loss module.
from minilmv2.loss import DistillationLoss


class MiniLM(nn.Module):
    """MiniLMv2 model with PCA-based initialization of student token embeddings.

    Arguments:
        teacher (nn.Module): The teacher model.
        student (nn.Module): The student model.
        L (int): The teacher layer (1-indexed) to distill.
        M (int): The student layer (1-indexed) used in distillation.
        relations (Dict[Tuple[int, int], float]): A dictionary mapping relation pairs (e.g., (1, 2) for query-key)
            to a corresponding weight.
        A_r (int): Number of relation heads.
        loss_type (str): 'kl' to use KL divergence or 'jsd' to use Jensen–Shannon divergence.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        L: int,
        M: int,
        relations: Dict[Tuple[int, int], float],
        A_r: int,
        loss_type: str = "kl",
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher.eval()
        self.student.train()
        self.L = L
        self.M = M
        self.relations = relations
        self.A_r = A_r
        self.loss_type = loss_type

        # Freeze teacher parameters.
        for param in self.teacher.parameters():
            param.requires_grad = False

        logging.warning(
            "Teacher model set to eval mode and gradients disabled. "
            "Remember to re-enable teacher training if necessary."
        )

        # Initialize student token embeddings from teacher using PCA.
        self.initialize_student_embeddings_from_teacher()

        # Instantiate the combined distillation loss module.
        self.dist_loss = DistillationLoss(
            loss_type=loss_type, reduction="batchmean", eps=1e-8
        )

    def initialize_student_embeddings_from_teacher(self):
        """Initialize the student's token embeddings using PCA on the teacher's embeddings.

        Teacher embeddings of shape [vocab_size, teacher_hidden_dim] are reduced to the student's embedding dimension.
        """
        teacher_embed = (
            self.teacher.embeddings.tok_embeddings.weight.detach().cpu().numpy()
        )
        vocab_size, teacher_hidden_dim = teacher_embed.shape
        student_hidden_dim = self.student.embeddings.tok_embeddings.embedding_dim

        logging.info(
            f"Teacher embedding shape: {teacher_embed.shape}. Student embedding dimension: {student_hidden_dim}"
        )

        if teacher_hidden_dim < student_hidden_dim:
            raise ValueError(
                "Teacher hidden dim is smaller than student hidden dim. PCA-based initialization expects teacher_hidden_dim > student_hidden_dim."
            )

        pca = PCA(n_components=student_hidden_dim)
        transformed_embeddings = pca.fit_transform(
            teacher_embed
        )  # (vocab_size, student_hidden_dim)
        transformed_embeddings = torch.tensor(
            transformed_embeddings,
            dtype=self.student.embeddings.tok_embeddings.weight.dtype,
        )
        with torch.no_grad():
            self.student.embeddings.tok_embeddings.weight.copy_(
                transformed_embeddings.to(
                    self.student.embeddings.tok_embeddings.weight.device
                )
            )
        logging.info(
            "Student token embeddings initialized from teacher embeddings using PCA."
        )

    def _get_relation_vectors(self, self_attn, prev_hidden, relation_head_size: int):
        """Compute query, key, and value relation vectors from an attention layer.

        Returns tensors of shape (batch_size, A_r, seq_length, relation_head_size).
        """
        qkv = self_attn.Wqkv(prev_hidden)  # (batch_size, seq_length, 3 * hidden_dim)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = self._transpose_for_scores_relation(q, relation_head_size)
        k = self._transpose_for_scores_relation(k, relation_head_size)
        v = self._transpose_for_scores_relation(v, relation_head_size)
        return (q, k, v)

    def _transpose_for_scores_relation(self, x: torch.Tensor, relation_head_size: int):
        """Reshape and permute tensor for relation head attention.

        Args:
            x (Tensor): (batch_size, seq_length, hidden_size)
            relation_head_size (int): Size per relation head.
        Returns:
            Tensor: (batch_size, A_r, seq_length, relation_head_size)
        """
        new_x_shape = [*x.size()[:-1], self.A_r, relation_head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def train(self, mode=True):
        """Override train() to ensure the teacher remains in eval mode."""
        super().train(mode)
        self.teacher.eval()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Perform a forward pass and compute the MiniLM distillation loss.

        Returns:
            A tuple containing the scalar loss.
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        teacher_outs = self.teacher(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        student_outs = self.student(
            **inputs, output_hidden_states=True, output_attentions=True
        )

        L = self.L  # teacher layer index (1-indexed)
        M = self.M  # student layer index (1-indexed)

        d_h_T = self.teacher.config.hidden_size
        d_h_S = self.student.config.hidden_size
        d_r_T = d_h_T // self.A_r
        d_r_S = d_h_S // self.A_r

        hidden_L_1_T = teacher_outs.hidden_states[L - 1]
        hidden_M_1_S = student_outs.hidden_states[M - 1]

        relation_vectors_T = self._get_relation_vectors(
            self.teacher.layers[L - 1].attn, hidden_L_1_T, d_r_T
        )
        relation_vectors_S = self._get_relation_vectors(
            self.student.layers[M - 1].attn, hidden_M_1_S, d_r_S
        )

        loss = 0.0
        # Loop over each relation pair.
        for relation_pair, weight in self.relations.items():
            m, n = relation_pair  # e.g., (1,2): 1->query, 2->key.
            A_L_T_scaleddot = torch.matmul(
                relation_vectors_T[m - 1], relation_vectors_T[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_T)
            A_M_S_scaleddot = torch.matmul(
                relation_vectors_S[m - 1], relation_vectors_S[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_S)

            # Use the combined loss module to compute the distillation loss.
            l_relation = self.dist_loss(
                A_L_T_scaleddot.detach(), A_M_S_scaleddot, inputs["attention_mask"]
            )
            loss += weight * l_relation

        return (loss,)
