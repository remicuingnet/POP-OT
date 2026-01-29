"""
Loss functions for overparameterized classification with slack variables
and optimal transport–based assignment.

Includes:
- Greedy assignment utility
- Standard cross-entropy wrapper
- Full overparameterization loss (SOP / SOP+ style)
- Partial overparameterization loss with Sinkhorn or LAP assignment
"""

# STL

# External
import lap
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project
from parse_config import ConfigParser

cross_entropy_val = nn.CrossEntropyLoss


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def greedy_assignment(S: torch.Tensor) -> torch.Tensor:
    """
    Convert a soft assignment matrix into a hard permutation matrix
    using a greedy maximum-weight matching.

    Args:
        S (torch.Tensor): (n, m) soft assignment matrix

    Returns:
        torch.Tensor: (n, m) hard assignment matrix
    """
    n = S.size(0)
    m = S.size(1)

    P = torch.zeros_like(S)

    assigned_rows = [False] * n
    assigned_cols = [False] * m
    nb_assigned = 0

    # Flatten and sort all entries by descending score
    _, indices = torch.sort(S.view(-1), descending=True)
    rows = indices // n
    cols = indices % n

    for r, c in zip(rows.tolist(), cols.tolist()):
        if not assigned_rows[r] and assigned_cols[c]:
            P[r, c] = 1.0
            assigned_rows[r] = True
            assigned_cols[c] = True
            nb_assigned += 1
            if nb_assigned == n:
                break

    return P


# ------------------------------------------------------------------------------
# Simple Cross-Entropy Wrapper
# ------------------------------------------------------------------------------


class cross_entropy(nn.Module):
    """
    Thin wrapper around standard cross-entropy loss.

    Also contains unused parameters (s, t) kept for compatibility
    with earlier experimental formulations.
    """

    def __init__(
        self,
        num_examp=50000,
        num_classes=10,
        *args,
        **kwargs,
    ):
        super(cross_entropy, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()

        # Legacy parameters (not used in forward)
        self.s = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.t = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.E = None

    def compute_loss(self):
        """
        Auxiliary loss based on s^2 - t^2 (legacy).
        """
        param_y = self.s**2 - self.t**2
        max_, _ = torch.max(param_y, dim=1)
        return torch.mean(max_)

    def forward(self, index, outputs, label):
        """
        Standard cross-entropy loss.

        Args:
            index: sample indices (unused)
            outputs: logits
            label: one-hot labels
        """
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
        else:
            output = outputs

        ce_loss = F.cross_entropy(output, label.argmax(dim=1))
        # return  ce_loss, None, 0
        return ce_loss


# ------------------------------------------------------------------------------
# Full Overparameterization Loss (SOP / SOP+)
# ------------------------------------------------------------------------------


class overparametrization_loss(nn.Module):
    """
    Overparameterized loss with per-sample slack variables (u, v): SOP/SOP+

    This loss modifies predictions using learned slack terms and
    enforces:
    - classification accuracy
    - optional balance regularization
    - optional consistency regularization
    """

    def __init__(
        self,
        num_examp,
        num_classes=10,
        ratio_consistency=0,
        ratio_balance=0,
        *args,
        **kwargs,
    ):
        super(overparametrization_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        # self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        # Slack parameters
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

         self.E = None

        self.init_param(
            mean=self.config["reparam_arch"]["args"]["mean"],
            std=self.config["reparam_arch"]["args"]["std"],
        )

    def init_param(self, mean=0.0, std=1e-8):
        """Initialize slack parameters."""
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        """
        Compute overparameterized classification loss.
        """
        # label = torch.zeros(len(label), self.config['num_classes']).cuda().scatter_(1, label.view(-1,1), 1)

        # Handle multiple forward passes (e.g., augmentation consistency)
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            ensembled_output = 0.5 * (output + output2).detach()

        else:
            output = outputs
            ensembled_output = output.detach()

        eps = 1e-4

        # Slack contributions
        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        self.E = U_square - V_square

        original_prediction = F.softmax(output, dim=1)

        # Modified prediction
        prediction = torch.clamp(
            original_prediction + U_square - V_square.detach(), min=eps
        )
        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        # Classification loss
        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

        label_one_hot = self.soft_to_hard(output.detach())

        # MSE loss (for V)
        MSE_loss = F.mse_loss(
            (label_one_hot + U_square - V_square), label, reduction="sum"
        ) / len(label)
        loss += MSE_loss

        # Balance regularization
        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(
                -(prior_distr * torch.log(avg_prediction)).sum(dim=0)
            )
            loss += self.ratio_balance * balance_kl

        # Consistency regularization
        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):

            consistency_loss = self.consistency_loss(index, output, output2)

            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, index, output1, output2):
        """ KL divergence between two predictions with different augmentation """

        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction="none")
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        """Convert soft predictions to one-hot labels."""

        with torch.no_grad():
            return (
                (torch.zeros(len(x), self.config["num_classes"]))
                .cuda()
                .scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
            )

# ------------------------------------------------------------------------------
# Partial Overparameterization Loss with OT / LAP Assignment
# ------------------------------------------------------------------------------
class partial_overparametrization_loss(nn.Module):
    """
    Partial Overparameterization Loss with Slack Assignment.

    This loss implements a *partial overparameterization* strategy for
    classification under noisy or ambiguous supervision. Instead of
    assigning a dedicated slack variable to every training example,
    it maintains a *fixed-size pool of slack variables* and dynamically
    assigns samples to slacks using optimal transport (Sinkhorn) or
    hard assignment (LAP).

    Key ideas:
    ----------
    • Each sample's prediction can be corrected by a learned slack term
      (U² - V²), improving robustness to noise and label errors.
    • Only a subset of slack variables are learnable; the rest are fixed
      (used to stabilize training under noise).
    • A memory bank stores recent predictions and labels to construct
      a cost matrix between samples and slacks.
    • Assignment between samples and slack variables is solved via:
        - Sinkhorn (soft assignment), or
        - LAPJV (hard assignment fallback).
    • Supports consistency regularization and class-balance regularization.

    This loss is particularly suited for:
    -----------------------------------
    • Learning with noisy labels
    • Overparameterized classification regimes
    • Semi-supervised or mixup-style training
    • Robust self-training using EMA predictions

    The implementation closely follows SOP/SOP+ style objectives with
    partial slack reuse.
    """
    def __init__(
        self,
        k_pred: int =5,
        noise_level=0,
        noise_level_error=0,
        num_classes=10,
        ratio_consistency=0,
        ratio_balance=0,
        num_slack=1024,
        sk_lambda=0.001,
        sk_num_iteration=10,
        hard_assignment=False,
        soft_to_hard_assignment=False,
        *args,
        **kwargs,
    ):
        """
        Partial overparameterization loss using a memory bank and
        assignment (Sinkhorn or LAP) between samples and slack variables.
    
        Args:
            k_pred (int):
                Number of top-k predicted classes that receive class-specific
                slack (V). Only these classes receive a "V slack" contribution .

            noise_level (float):
                Fraction of slack variables treated as noisy / learnable.
                Must be in [0, 1].

            noise_level_error (float):
                Additional noise percentage added to noise_level
                (noise_level_error / 100).

            num_classes (int):
                Number of classification classes.

            ratio_consistency (float):
                Weight for consistency regularization between multiple
                forward passes (e.g., augmentations).

            ratio_balance (float):
                Weight for class-balance KL regularization.

            num_slack (int):
                Total number of slack slots maintained in the memory bank.

            sk_lambda (float):
                Entropic regularization strength for Sinkhorn optimal transport.

            sk_num_iteration (int):
                Maximum number of Sinkhorn iterations.

            hard_assignment (bool):
                If True, bypass Sinkhorn and use hard assignment via LAPJV.

            soft_to_hard_assignment (bool):
                If True, converts Sinkhorn soft assignment into a hard
                permutation matrix.

        """
        super().__init__()
        device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        # self.use_cuda = torch.cuda.is_available()


        # Effective noise level (clipped to [0, 1])
        self.noise_level = float(noise_level + noise_level_error / 100)  # 0.40208
        self.noise_level = max(0.0, min(1.0, self.noise_level))

        print(f"loss noise level: {self.noise_level}")

        # Total number of slack slots
        self.num_slack = num_slack  # 2048

        # Number of class-specific slack entries per sample
        self.k = k_pred 
        
        # Split slack pool into noisy (learnable) and fixed (zero) slacks based on noise level
        noise_instances = int(round(self.noise_level * self.num_slack))
        self.num_slack_list = (
            noise_instances,
            self.num_slack - noise_instances,
        )  # num_examp

        self.num_slack = sum(self.num_slack_list)

        # Sinkhorn parameters
        self.sk_lambda = sk_lambda
        self.sk_num_iteration = sk_num_iteration

        # Assignment strategy
        self.hard_assignment = hard_assignment
        self.soft_to_hard_assignment = soft_to_hard_assignment

        # Regularization weights
        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        # Learnable slack variables (positive via squaring)
        # u: class-agnostic slack
        # v: class-specific slack (top-k only)
        self.u = nn.Parameter(
            torch.empty(self.num_slack_list[0], 1, dtype=torch.float32)
        )
        self.v = nn.Parameter(
            torch.empty(self.num_slack_list[0], self.k, dtype=torch.float32)
        )

        # Fixed zero slack variables (non-learnable)
        self.u2 = torch.zeros(
            self.num_slack_list[1], 1, dtype=torch.float32, device=device
        )
        self.v2 = torch.zeros(
            self.num_slack_list[1], self.k, dtype=torch.float32, device=device
        )
        
        # Training state
        self.epoch = 1
        
        # Memory bank (filled during training)
        self.features_memory = None  # (num_slack, num_classes)
        self.label_memory = None     # (num_slack, num_classes)

        # Initialize learnable slack parameters
        self.init_param(
            mean=self.config["reparam_arch"]["args"]["mean"],
            std=self.config["reparam_arch"]["args"]["std"],
        )

    def init_param(self, mean=0.0, std=1e-8):
        """Initialize learnable slack parameters."""
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label, ema_output=None):
        """
        Compute the partial overparameterization loss.

        Args:
            index (LongTensor):
                Dataset indices for the current batch. Shape: (B,)

            outputs (FloatTensor):
                Model logits.
                Shape:
                    - (B, C): single forward
                    - (3B, C): (output, output2, output_mixup)

            label (FloatTensor):
                One-hot or soft labels. Shape: (B, C)

            ema_output (FloatTensor, optional):
                Logits from an EMA teacher model. Shape: (B, C) or (2B, C)

        Returns:
            torch.Tensor:
                Scalar loss value.
        """
        eps = 1e-4

        # Guard against NaNs in slack parameters
        if torch.any(torch.isnan(self.u.data)):
            raise ValueError("NaN in self.u")
        self.u.data[torch.isnan(self.u.data)] = 0

        if torch.any(torch.isnan(self.u.data)):
            raise ValueError("NaN in self.v")
        self.v.data[torch.isnan(self.v.data)] = 0

        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)

            ensembled_output = 0.5 * (output + output2).detach()

        else:
            output = outputs
            ensembled_output = output.detach()

        original_prediction = F.softmax(output, dim=1)
        ema_prediction = original_prediction
        if ema_output is not None:
            if len(outputs) > len(index):
                ema_output, _ = torch.chunk(ema_output, 2)
            ema_prediction = F.softmax(ema_output, dim=1)

        # Update memory bank
        if self.features_memory is None:
            self.features_memory = ema_prediction.detach().clone()
            self.label_memory = label.detach().clone()

        else:
            self.features_memory = torch.concatenate(
                (ema_prediction.detach().clone(), self.features_memory),
                dim=0,
            )
            self.label_memory = torch.concatenate(
                (label.detach().clone(), self.label_memory),
                dim=0,
            )

            self.features_memory = self.features_memory[: self.num_slack]
            self.label_memory = self.label_memory[: self.num_slack]

        # Warm-up: no assignment until memory is full
        if self.features_memory.shape[0] < self.num_slack:
            loss = torch.mean(
                -torch.sum((label) * torch.log(original_prediction), dim=-1)
            )
            prediction = original_prediction
        else:
            # ------------------------------------------------------------------
            # Build cost matrix between samples in memory and slack variables
            # ------------------------------------------------------------------
            with torch.no_grad():
                # Compute u^2 and v^2
                
                # Concatenate learnable and fixed slack variables
                # u2: (N, 1), v2: (N, k)
                u2 = torch.cat((self.u.detach(), self.u2))
                u2 = u2 * u2                                # enforce positivity
                u2_exp = u2.unsqueeze(0)                    # (1, N, 1)

                v2 = torch.cat((self.v.detach(), self.v2))
                v2 = v2 * v2                                # (N, k)

                # Memory predictions and labels
                f = self.features_memory.detach()
                f_exp = f  # (B, C)
                label_exp = self.label_memory.detach().unsqueeze(1)  # (B, 1, C)

                _, topk_indices = torch.topk(f, k=self.k, dim=1)  # [B, k]
                label_one_hot = self.soft_to_hard(f)
                label_one_hot_exp = label_one_hot.unsqueeze(1)  # (B, 1, C)
                B = f_exp.shape[0]
                N = v2.shape[0]

                # 1. Broadcast pred_exp -> [B, N, C]
                pred_exp2 = f_exp.unsqueeze(1).expand(-1, N, -1).clone()

                # 2. Expand topk_indices -> [B, N, k]
                topk_idx_expanded = topk_indices.unsqueeze(1).expand(-1, N, -1)

                # 3. Expand v -> [B, N, k]
                v_expanded = v2.unsqueeze(0).expand(B, -1, -1)  # [B, N, k]
                v2_exp = torch.zeros_like(pred_exp2)

                # 4. Scatter into v2_exp (to add be added in pred_exp2) [B, N, C]
                v2_exp.scatter_add_(2, topk_idx_expanded, v_expanded)

                U_square = u2_exp * label_exp
                V_square = v2_exp * (1 - label_exp)

                prediction = torch.clamp(
                    pred_exp2 + U_square - V_square,
                    min=eps,
                )

                prediction = F.normalize(prediction, p=1, eps=eps, dim=1)

                prediction = torch.clamp(prediction, min=eps, max=1.0)

                MSE_loss = (
                    label_one_hot_exp
                    - label_exp
                    + U_square  # * (1 - label_one_hot_exp)
                    - V_square  # * label_one_hot_exp
                )  # (B, N, C)
                # Compute squared L2 norm along last dim
                MSE_loss = (MSE_loss * MSE_loss).sum(dim=2)  # (B, N)
                loss = -torch.sum((label_exp) * torch.log(prediction), dim=2)
                loss += MSE_loss

                dist_matrix = loss
                if torch.any(torch.isnan(dist_matrix)):
                    raise ValueError("NaN in dist_matrix")

                if torch.any(torch.isnan(dist_matrix)) or torch.any(
                    torch.isinf(dist_matrix)
                ):
                    print("invalid numeric entries")
                    for var, name in [
                        (f_exp, "pred"),
                        (label_exp, "label"),
                        (u2_exp, "u2"),
                        (v2_exp, "v2"),
                    ]:
                        if torch.any(torch.isnan(var)):
                            print(f"NaN in {name}")
                        if torch.any(torch.isinf(var)):
                            print(f"Inf in {name}")

            # ------------------------------------------------------------------
            # Assignment: Sinkhorn (soft) or LAP (hard)
            # ------------------------------------------------------------------
            soft_has_failed = False
            if not (self.hard_assignment or soft_has_failed):
                source_distribution = torch.ones(
                    dist_matrix.size(0)
                ) / dist_matrix.size(0)
                source_distribution = source_distribution.to(output.device)
                target_distribution = torch.ones(
                    dist_matrix.size(0)
                ) / dist_matrix.size(0)
                target_distribution = target_distribution.to(output.device)

                # Sinkhorn optimal transport
                assignment_weights = ot.bregman.sinkhorn_log(
                    source_distribution,
                    target_distribution,
                    dist_matrix,
                    reg=self.sk_lambda,
                    numItermax=self.sk_num_iteration,
                )

                if torch.any(torch.isnan(assignment_weights)):
                    print("NaN in Sinkhorn")
                    soft_has_failed = True
                else:

                    # Normalize rows
                    assignment_weights_init = assignment_weights
                    assignment_weights = assignment_weights / assignment_weights.sum(
                        dim=1, keepdim=True
                    )

                    # Optionally convert soft assignment to hard permutation
                    if self.soft_to_hard_assignment:
                        assignment_weights = greedy_assignment(assignment_weights)

                    # Apply assigned slack to current batch
                    assignment_weights = assignment_weights[: output.shape[0], :]
                    u2 = torch.cat((self.u, self.u2))

                    v2 = torch.cat((self.v, self.v2))
                    U_square = assignment_weights @ (u2**2)  # [B, n classes]
                    V_square_2 = assignment_weights @ (v2**2)  # [B, n classes]

                    # Apply slacks only to appropriate classes
                    U_square = U_square * label
                    V_square = torch.zeros_like(original_prediction)
                    V_square.scatter_(1, topk_indices[: output.shape[0]], V_square_2)

                    if torch.any(torch.isnan(U_square)):
                        raise ValueError(
                            f"NaN in U_square {assignment_weights_init.sum(dim=1)}"
                        )
                    if torch.any(torch.isnan(V_square)):
                        raise ValueError("NaN in V_square")
                    # print(assignment_weights.shape, u.shape, v.shape)

            # ------------------------------------------------------------------
            # Hard assignment fallback (LAPJV)
            # ------------------------------------------------------------------
            if soft_has_failed or self.hard_assignment:
                cost = dist_matrix.cpu().numpy()

                cost, x, y = lap.lapjv(cost)
                col_ind = x

                col_ind = col_ind[: output.shape[0]]
                u2 = torch.cat((self.u, self.u2))
                v2 = torch.cat((self.v, self.v2))
                U_square = (u2[col_ind] ** 2) * label
                V_square = (v2[col_ind] ** 2) * (1 - label)  # * label_one_hot

            # ------------------------------------------------------------------
            # Final corrected prediction and loss
            # ------------------------------------------------------------------
            U_square = torch.clamp(U_square, 0, 1)
            V_square = torch.clamp(V_square, 0, 1)

            E = U_square - V_square

            self.E = E

            prediction = torch.clamp(
                original_prediction + U_square - V_square.detach(),
                min=eps,
            )

            prediction = F.normalize(prediction, p=1, eps=eps)

            prediction = torch.clamp(prediction, min=eps, max=1.0)


            loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

            # Hard labels for MSE regularization
            label_one_hot = self.soft_to_hard(output.detach())
            MSE_loss = F.mse_loss(
                (label_one_hot + U_square - V_square), label, reduction="sum"
            ) / len(label)
            loss += MSE_loss

        # ------------------------------------------------------------------
        # Class-balance regularization
        # ------------------------------------------------------------------
        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(
                -(prior_distr * torch.log(avg_prediction)).sum(dim=0)
            )

            loss += self.ratio_balance * balance_kl
            if torch.any(torch.isnan(loss)):
                raise ValueError("NaN in KL loss")


        # ------------------------------------------------------------------
        # Consistency regularization between two forward passes
        # ------------------------------------------------------------------
        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):

            consistency_loss = self.consistency_loss(index, output, output2)
            if torch.any(torch.isnan(consistency_loss)):
                raise ValueError("NaN in consistency_loss")
            loss += self.ratio_consistency * torch.mean(consistency_loss)

        if torch.any(torch.isnan(loss)):
            raise ValueError("NaN in loss")

        return loss

    def consistency_loss(self, index, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction="none")
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (
                (torch.zeros(len(x), self.config["num_classes"]))
                .cuda()
                .scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
            )

    def set_epoch(self, epoch):
        self.epoch = epoch
