"""Probability calibration.

Neural nets trained with cross-entropy are usually over-confident: "95%" does not
mean right 95% of the time. Our decision policy (auto-accept above 90%) only works
if probabilities are honest, so we fit a single temperature T on the validation set
(Guo et al., 2017, "On Calibration of Modern Neural Networks"):

    p = softmax(logits / T)

T > 1 softens over-confident predictions. It does not change the argmax, so
accuracy is unaffected; only the confidence numbers become trustworthy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    # clone(): logits may be inference-mode tensors, which autograd refuses to use
    logits = logits.detach().float().clone()
    labels = labels.detach().long().clone()
    log_t = torch.zeros(1, requires_grad=True)   # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return min(max(t, 0.05), 20.0)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Weighted average gap between confidence and accuracy over confidence bins."""
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece)
