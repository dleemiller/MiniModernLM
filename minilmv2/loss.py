import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Computes a loss for aligning teacher and student relation logits,
    either using KL divergence or Jensen–Shannon divergence, and scales the loss
    by the effective sequence length (derived from the attention mask).

    The inputs are assumed to have shape:
        teacher_logits, student_logits: (batch, A_r, seq_len, seq_len)
        attention_mask: (batch, seq_len) -- with 1 for valid tokens, 0 for padded tokens.

    The loss is computed sample-wise by slicing each batch element to its effective sequence length.
    For a single sample with effective sequence length L_eff:
        - KL Loss:
            loss_sample = KL( log_softmax(student_logits) || log_softmax(teacher_logits) )
            computed on a reshaped tensor of shape (A_r * L_eff, L_eff),
        - JS Loss:
            first compute P = softmax(teacher_logits) and Q = softmax(student_logits),
            then M = 0.5 * (P + Q) and compute:
                JS(P || Q) = 0.5 * [KL(P || M) + KL(Q || M)]
            computed over the same dimensions.
    Finally, the loss for each sample is divided by its effective sequence length, and the average is taken over the batch.

    Args:
        loss_type (str): Either 'kl' or 'jsd'. If 'kl', use KL divergence; if 'jsd', use Jensen–Shannon divergence.
        reduction (str): Reduction mode passed to KLDivLoss (default "batchmean").
        eps (float): Small constant for numerical stability (default 1e-8).
    """

    def __init__(
        self, loss_type: str = "kl", reduction: str = "batchmean", eps: float = 1e-8
    ):
        super().__init__()
        if loss_type not in ("kl", "jsd"):
            raise ValueError("loss_type must be either 'kl' or 'jsd'")
        self.loss_type = loss_type
        self.eps = eps
        self.reduction = reduction
        # Instantiate the PyTorch KL divergence loss module (used in both cases).
        self.kl_loss_fn = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            teacher_logits (Tensor): Teacher relation logits, shape (batch, A_r, seq_len, seq_len)
            student_logits (Tensor): Student relation logits, same shape as teacher_logits.
            attention_mask (Tensor): Binary mask of shape (batch, seq_len), with 1 for valid tokens.
        Returns:
            Tensor: Scalar loss value.
        """
        batch_size = teacher_logits.size(0)
        loss_sum = 0.0

        # Process each sample in the batch individually
        for b in range(batch_size):
            # Compute effective sequence length (number of valid tokens)
            seq_len = int(attention_mask[b].sum().item())
            # Slice logits up to effective sequence length.
            t_logits = teacher_logits[
                b, :, :seq_len, :seq_len
            ]  # shape: (A_r, seq_len, seq_len)
            s_logits = student_logits[
                b, :, :seq_len, :seq_len
            ]  # shape: (A_r, seq_len, seq_len)

            if self.loss_type == "kl":
                # Compute log probabilities
                t_log_prob = F.log_softmax(t_logits, dim=-1)
                s_log_prob = F.log_softmax(s_logits, dim=-1)
                # Reshape into (A_r * seq_len, seq_len) for KLDivLoss.
                t_log_prob = t_log_prob.reshape(-1, seq_len)
                s_log_prob = s_log_prob.reshape(-1, seq_len)
                loss_sample = self.kl_loss_fn(s_log_prob, t_log_prob)
            else:  # loss_type == "jsd"
                # Compute probabilities
                t_prob = F.softmax(t_logits, dim=-1)
                s_prob = F.softmax(s_logits, dim=-1)
                # Compute the average distribution M.
                M = 0.5 * (t_prob + s_prob)
                # Compute log probabilities with numerical stability.
                t_log_prob = torch.log(t_prob + self.eps)
                s_log_prob = torch.log(s_prob + self.eps)
                M_log = torch.log(M + self.eps)
                # Compute KL divergences per head and per token (sum over last dimension)
                kl_teacher = (t_prob * (t_log_prob - M_log)).sum(dim=-1)
                kl_student = (s_prob * (s_log_prob - M_log)).sum(dim=-1)
                # Average the two KL values to get JS divergence per head and token.
                js_div = 0.5 * (kl_teacher + kl_student)
                # Average over the A_r heads and sequence dimension.
                loss_sample = js_div.mean()

            # Normalize by the effective sequence length.
            loss_sum += loss_sample / seq_len

        # Return the average loss over the batch.
        loss_final = loss_sum / batch_size
        return loss_final
