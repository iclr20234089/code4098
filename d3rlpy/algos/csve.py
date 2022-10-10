from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from ..dynamics import DynamicsBase
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from .base import AlgoBase
from .torch.CSVE_impl import CSVEImpl


class CSVE(AlgoBase):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _value_encoder_factory: EncoderFactory
    _tau: float
    _n_critics: int
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _use_gpu: Optional[Device]
    _impl: Optional[CSVEImpl]
    _dynamics: Optional[DynamicsBase]
    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        value_encoder_factory: EncoderArg = "default",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        expectile: float = 1,
        weight_temp: float = 3.0,
        initial_alpha: float = 10.0,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 5.0,
        max_weight: float = 100.0,
        bc=1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[CSVEImpl] = None,
        dynamics,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._value_encoder_factory = check_encoder(value_encoder_factory)
        self._tau = tau
        self._bc = bc
        self._n_critics = n_critics
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._conservative_weight = conservative_weight
        self._alpha_threshold = alpha_threshold
        self._dynamics = dynamics
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CSVEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            value_encoder_factory=self._value_encoder_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            expectile=self._expectile,
            weight_temp=self._weight_temp,
            max_weight=self._max_weight,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            dynamics=self._dynamics,
            alpha_learning_rate=self._alpha_learning_rate,
            alpha_threshold=self._alpha_threshold,
            conservative_weight=self._conservative_weight,
            alpha_optim_factory=self._alpha_optim_factory,
            initial_alpha=self._initial_alpha,
            bc=self._bc,

        )
        self._impl.build()


    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}
        if self._alpha_learning_rate > 0:
            alpha_loss, alpha,loss = self._impl.update_alpha(batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha, "diff" : loss})

        critic_loss, value_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss, "value_loss": value_loss})

        actor_loss,loss1,loss2,adv2 = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss,"loss1":loss1,"loss2":loss2, "adv":adv2})

        self._impl.update_critic_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
