from abc import ABCMeta, abstractmethod
from typing import Optional

import torch

from ..encoders import Encoder


class VFunction(metaclass=ABCMeta):
    @abstractmethod
    def compute_error(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass


class ContinuousVFunction(VFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def encoder(self) -> Encoder:
        pass
