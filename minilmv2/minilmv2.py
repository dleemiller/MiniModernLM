import logging
import math
from typing import Dict, Tuple

import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA


class MiniLM(nn.Module):
    """MiniLMv2 model with PCA-based initialization of student token embeddings.

    Arguments:
        teacher (nn.Module): the teacher model.
        student (nn.Module): the student model.
        L (int): Lth layer of the teacher model to distill.
        M (int): Number of layers of the student model used in distillation.
        relations (Dict[Tuple[int, int], float]): A dictionary of self-attention relation pairs and weights.
        A_r (int): Number of relation heads.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        L: int,
        M: int,
        relations: Dict[Tuple[int, int], float],
        A_r: int,
    ):
        """Initialize a MiniLMv2 model."""
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.teacher.eval()
        self.student.train()
        self.L = L
        self.M = M
        self.relations = relations  # relation weights as hyperparameters
        self.A_r = A_r

        # Do not update teacher parameter
        for param in self.teacher.parameters():
            param.requires_grad = False

        logging.warning(
            "Setting teacher model to eval mode and disabling gradient update for MiniLM training. "
            "You must manually reset it to train mode and enable gradient update if you wish to continue updating "
            "the teacher after distillation."
        )

        # Initialize student embeddings using teacher embeddings with PCA.
        self.initialize_student_embeddings_from_teacher()

    def initialize_student_embeddings_from_teacher(self):
        """Initialize the student's token embeddings using PCA on the teacher's token embeddings.

        The teacher's embeddings (shape: [vocab_size, teacher_hidden_dim])
        are reduced to the student's embedding dimension.
        """
        # 1. Extract teacher embeddings. (Assuming attribute naming as in your architecture.)
        teacher_embed = (
            self.teacher.embeddings.tok_embeddings.weight.detach().cpu().numpy()
        )
        vocab_size, teacher_hidden_dim = teacher_embed.shape

        # 2. Get student's embedding dimension.
        student_hidden_dim = self.student.embeddings.tok_embeddings.embedding_dim

        # Log dimensions.
        logging.info(
            f"Teacher embedding shape: {teacher_embed.shape}. Student embedding dimension: {student_hidden_dim}"
        )

        # 3. Compute PCA if teacher_hidden_dim is larger than student_hidden_dim.
        if teacher_hidden_dim < student_hidden_dim:
            raise ValueError(
                "Teacher hidden dim is smaller than student hidden dim. PCA-based initialization "
                "expects teacher hidden dimension to be larger."
            )

        pca = PCA(n_components=student_hidden_dim)
        transformed_embeddings = pca.fit_transform(
            teacher_embed
        )  # shape: (vocab_size, student_hidden_dim)

        # 4. Convert to torch tensor with same dtype as student embeddings.
        transformed_embeddings = torch.tensor(
            transformed_embeddings,
            dtype=self.student.embeddings.tok_embeddings.weight.dtype,
        )

        # 5. Copy the PCA-transformed weights into the student's token embeddings.
        with torch.no_grad():
            self.student.embeddings.tok_embeddings.weight.copy_(
                transformed_embeddings.to(
                    self.student.embeddings.tok_embeddings.weight.device
                )
            )
        logging.info(
            "Student token embeddings have been initialized from teacher embeddings using PCA."
        )

    def _get_relation_vectors(self, self_attn, prev_hidden, relation_head_size: int):
        """Get query, key, and value relation vectors from an attention layer.

        Returns vectors of shape (batch_size, A_r, seq_length, relation_head_size).
        """
        # Project Q, K, V using Wqkv
        qkv = self_attn.Wqkv(
            prev_hidden
        )  # shape: (batch_size, seq_length, 3 * hidden_dim)

        # Split into query, key, and value (assuming equal division)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Transpose for relation heads.
        q = self._transpose_for_scores_relation(q, relation_head_size)
        k = self._transpose_for_scores_relation(k, relation_head_size)
        v = self._transpose_for_scores_relation(v, relation_head_size)

        return q, k, v

    def _transpose_for_scores_relation(self, x: torch.Tensor, relation_head_size: int):
        """Transpose and reshape tensor for relation head attention.

        Args:
            x (Tensor): shape (batch_size, seq_length, hidden_size)
            relation_head_size (int): size per relation head.
        Returns:
            Tensor: shape (batch_size, A_r, seq_length, relation_head_size)
        """
        new_x_shape = [*x.size()[:-1], self.A_r, relation_head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_kl_loss(
        self, rel_T: torch.Tensor, rel_S: torch.Tensor, attention_mask: torch.Tensor
    ):
        loss = 0.0
        batch_size = attention_mask.shape[0]
        seq_lengths = attention_mask.sum(-1).tolist()

        for b in range(batch_size):
            cur_seq_len = seq_lengths[b]

            teacher_logits = rel_T[b, :, :cur_seq_len, :cur_seq_len]
            student_logits = rel_S[b, :, :cur_seq_len, :cur_seq_len]

            # Compute distributions in log-space.
            teacher_log_dist = torch.nn.functional.log_softmax(teacher_logits, dim=-1)
            student_log_dist = torch.nn.functional.log_softmax(student_logits, dim=-1)

            # Compute KL divergence loss with batchmean.
            # Note: batchmean averages only by batch size.
            loss_batch = self.kl_loss_fn(
                student_log_dist.reshape(-1, cur_seq_len),
                teacher_log_dist.reshape(-1, cur_seq_len),
            )

            loss += loss_batch / cur_seq_len

        return loss

    def train(self, mode=True):
        """Override the train method to ensure the teacher remains in eval mode."""
        super().train(mode)
        self.teacher.eval()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Perform a forward pass and compute MiniLM distillation loss.

        Returns:
            A tuple containing the loss.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        teacher_outs = self.teacher(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        student_outs = self.student(
            **inputs, output_hidden_states=True, output_attentions=True
        )

        L = self.L  # teacher layer index (1-indexed)
        M = self.M  # student layer index (1-indexed)

        d_h_T = self.teacher.config.hidden_size  # teacher hidden size
        d_h_S = self.student.config.hidden_size  # student hidden size
        d_r_T = d_h_T // self.A_r  # teacher relation head size
        d_r_S = d_h_S // self.A_r  # student relation head size

        # Get hidden states for the second last layer (L-1 for teacher, M-1 for student)
        hidden_L_1_T = teacher_outs.hidden_states[L - 1]
        hidden_M_1_S = student_outs.hidden_states[M - 1]

        # Get relation vectors: each returns (q, k, v) of shape (batch_size, A_r, seq_len, relation_head_size)
        relation_vectors_T = self._get_relation_vectors(
            self.teacher.layers[L - 1].attn, hidden_L_1_T, d_r_T
        )
        relation_vectors_S = self._get_relation_vectors(
            self.student.layers[M - 1].attn, hidden_M_1_S, d_r_S
        )

        loss = 0.0
        # Loop over each relation pair and its weight.
        for relation_pair, weight in self.relations.items():
            # relation_pair: (m, n) where 1->Query, 2->Key, 3->Value.
            m, n = relation_pair

            # Compute scaled dot products for teacher and student (formulas (7) and (8)).
            A_L_T_scaleddot = torch.matmul(
                relation_vectors_T[m - 1], relation_vectors_T[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_T)
            A_M_S_scaleddot = torch.matmul(
                relation_vectors_S[m - 1], relation_vectors_S[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_S)

            # Compute the KL divergence loss for this relation pair.
            l_relation = self._get_kl_loss(
                A_L_T_scaleddot.detach(), A_M_S_scaleddot, inputs["attention_mask"]
            )

            # Weight and accumulate the loss.
            loss += weight * l_relation

        return (loss,)
