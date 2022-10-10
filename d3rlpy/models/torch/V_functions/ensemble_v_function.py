from typing import List, Optional, Union, cast

import torch
from torch import nn
import torch.nn.functional as F
from .base import ContinuousVFunction


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "pessimistic", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    elif reduction == "pessimistic":
        return y.mean(dim=dim)-0.1*y.std(dim=dim)
    elif reduction == "pessi2":
        return y-0.1*y.std(dim=dim)

    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleVFunction(nn.Module):  # type: ignore
    _v_funcs: nn.ModuleList

    def __init__(
        self,
        v_funcs: Union[ List[ContinuousVFunction]],
    ):
        super().__init__()
        self._v_funcs = nn.ModuleList(v_funcs)

    def compute_error(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        q_tp: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        if use_independent_target:
            assert q_tp.ndim == 3
        else:
            assert q_tp.ndim == 2

        td_sum = torch.tensor(0.0, dtype=torch.float32, device=observations.device)
        for i, v_func in enumerate(self._v_funcs):
            if use_independent_target:
                target = q_tp[i]
            else:
                target = q_tp

            loss = v_func.compute_error(
                                observations=observations,
                rewards=rewards,
                target=target,
                terminals=terminals,
                gamma=gamma,
                reduction="none",
            )
            # print(loss.mean())

            td_sum += loss.mean()
        return td_sum

    def _compute_target(
        self,
        x: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for v_func in self._v_funcs:
            target = v_func.compute_target(x)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)
        return _reduce_ensemble(values, reduction)

    @property
    def v_funcs(self) -> nn.ModuleList:
        return self._v_funcs



# class PessiVFunction(nn.Module):  # type: ignore
#     _v_funcs: nn.ModuleList
#     _bootstrap: bool
#
#     def __init__(
#         self,
#         v_funcs: Union[ List[ContinuousVFunction]],
#         bootstrap: bool = False,
#     ):
#         super().__init__()
#         self._v_funcs = nn.ModuleList(v_funcs)
#         self._bootstrap = bootstrap and len(v_funcs) > 1
#
#     def compute_error(
#         self,
#         obs_t: torch.Tensor,
#         rew_tp1: torch.Tensor,
#         q_tp1: torch.Tensor,
#         ter_tp1: torch.Tensor,
#         gamma: float = 0.99,
#         use_independent_target: bool = False,
#         masks: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if use_independent_target:
#             assert q_tp1.ndim == 3
#         else:
#             assert q_tp1.ndim == 2
#
#         if self._bootstrap and masks is not None:
#             assert masks.shape == (len(self._v_funcs), obs_t.shape[0], 1,), (
#                 "Invalid mask shape is detected. "
#                 f"mask_size must be {len(self._v_funcs)}."
#             )
#
#         td_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
#         for i, v_func in enumerate(self._v_funcs):
#             if use_independent_target:
#                 target = q_tp1[i]
#             else:
#                 target = q_tp1
#
#             loss = v_func.compute_error(
#                 obs_t, rew_tp1, target, ter_tp1, gamma, reduction="none"
#             )
#             # print(loss.mean())
#             if self._bootstrap:
#                 if masks is None:
#                     mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
#                 else:
#                     mask = masks[i]
#                 loss *= mask.float()
#                 td_sum += loss.sum() / (mask.sum().float() + 1e-10)
#             else:
#                 td_sum += loss.mean()
#
#         return td_sum
#
#     def _compute_target(
#         self,
#         x: torch.Tensor,
#         reduction: str = "min",
#         lam: float = 0.75,
#     ) -> torch.Tensor:
#         values_list: List[torch.Tensor] = []
#         for v_func in self._v_funcs:
#             target = v_func.compute_target(x)
#             values_list.append(target.reshape(1, x.shape[0], -1))
#
#         values = torch.cat(values_list, dim=0)
#         return _reduce_ensemble(values, reduction)
#
#     @property
#     def v_funcs(self) -> nn.ModuleList:
#         return self._v_funcs
#
#     @property
#     def bootstrap(self) -> bool:
#         return self._bootstrap
class EnsembleContinuousVFunction(EnsembleVFunction):
    def forward(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for v_func in self._v_funcs:
            values.append(v_func(x).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor,  reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x,reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        reduction: str = "pessimistic",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, reduction, lam)
