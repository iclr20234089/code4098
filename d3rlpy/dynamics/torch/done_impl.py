from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_done_model

from ...models.optimizers import OptimizerFactory
from ...models.torch import ProbabilisticEnsembleDynamicsModel
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .base import TorchDoneImplBase


class DoneImpl(TorchDoneImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _use_gpu: Optional[Device]
    _dynamics: Optional[ProbabilisticEnsembleDynamicsModel]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        use_gpu: Optional[Device],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory


        self._use_gpu = use_gpu

        # initialized in build
        self._done = None
        self._optim = None

    def build(self) -> None:
        self._build_done()

        self.to_cpu()
        if self._use_gpu:
            self.to_gpu(self._use_gpu)

        self._build_optim()

    def _build_done(self) -> None:
        self._done = create_done_model(
            self._observation_shape,
            self._action_size,
        )

    def _build_optim(self) -> None:
        assert self._done is not None
        self._optim = self._optim_factory.create(
            self._done.parameters(), lr=self._learning_rate
        )

    def _predict(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) ->torch.Tensor:
        assert self._done is not None
        return self._done.predict_done(
            x,
            action,
        )
    # def _predict_sigmoid(
    #     self,
    #     x: torch.Tensor,
    #     action: torch.Tensor,
    # ) -> torch.Tensor:
    #     assert self._done is not None
    #     return self._done.predict_done_sigmoid(
    #         x,
    #         action,
    #     )
    @train_api
    @torch_api()
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._done is not None
        assert self._optim is not None

        loss = self._done.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions,
            done_tp1=batch.terminals,
        )

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()
