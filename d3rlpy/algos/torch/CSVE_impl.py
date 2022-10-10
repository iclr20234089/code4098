
import math
from typing import Optional, Sequence

import numpy as np
import torch

from ...gpu import Device
from ...models.builders import (
    create_non_squashed_normal_policy,
    create_value_function,
    create_parameter,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import MeanQFunctionFactory
from ...models.torch import NonSquashedNormalPolicy, ValueFunction
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .ddpg_impl import DDPGBaseImpl


class CSVEImpl(DDPGBaseImpl):
    _policy: Optional[NonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _conservative_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        alpha_threshold: float,
        conservative_weight: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        alpha_optim_factory: OptimizerFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        dynamics,
        bc,
        initial_alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=MeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,

        )
        self._expectile = expectile
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._conservative_weight = conservative_weight
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None
        self._bc =bc
        self.dynamics = dynamics
        if self._use_gpu:
            self.dynamics.impl.to_gpu(self._use_gpu)

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_value_function(
            self._observation_shape, self._value_encoder_factory
        )

    def build(self) -> None:
        self._build_alpha()
        super().build()
        self._build_alpha_optim()

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        self._log_alpha = create_parameter((1, 1), initial_val)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            self._log_alpha.parameters(), lr=self._alpha_learning_rate
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim1 = self._critic_optim_factory.create(
            q_func_params , lr=self._critic_learning_rate
        )
        self._critic_optim2 = self._critic_optim_factory.create(
            v_func_params , lr=self._critic_learning_rate
        )
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params+v_func_params , lr=self._critic_learning_rate
        )
    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )



    @train_api
    @torch_api()
    def update_alpha(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._value_func.eval()

        self._alpha_optim.zero_grad()

        # the original implementation does scale the loss value
        loss,diff = self.compute_conservative_value_loss(
            batch
        )

        loss.backward()
        self._alpha_optim.step()

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha,diff

    def compute_conservative_value_loss(
        self, batch:TorchMiniBatch
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._value_func is not None
        with torch.no_grad():
            policy_actions= self._policy.best_action(
                batch.observations)
            s_ood, _, _ = self.dynamics.predict_gpu(batch.observations, policy_actions, True)
            s_ood = s_ood.clamp(-10, 10)
        ood_values = self._value_func(s_ood)
        ## uncomment this statement if the training is not stable
        #ood_values = ood_values.clamp(-1500, 1500)

        data_values = self._value_func(batch.next_observations)
        #data_values = data_values.clamp(-1500, 1500)
        loss = ood_values.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]
        # clipped_alpha = 10
    # return clipped_alpha * (scaled_loss - self._alpha_threshold)
        return -clipped_alpha*(scaled_loss-self._alpha_threshold), loss.cpu().detach().numpy()
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            return self._value_func(batch.next_observations)
    #
    # def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
    #     assert self._policy
    #
    #     # compute log probability
    #     dist = self._policy.dist(batch.observations)
    #     log_probs = dist.log_prob(batch.actions)
    #
    #     # compute weight
    #     with torch.no_grad():
    #         weight = self._compute_weight(batch)
    #
    #     return -(weight * log_probs).mean()
    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss1,loss2,adv2 = self.compute_actor_loss(batch)

        loss = loss1+loss2
        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy(),loss1.cpu().detach().numpy(),loss2.cpu().detach().numpy(),adv2.cpu().detach().numpy()
    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        #q_t = q_t.clamp(-1500, 1500)
        v_t = self._value_func(batch.observations)
        #v_t = v_t.clamp(-1500, 1500)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    # def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
    #     assert self._targ_q_func
    #     assert self._value_func
    #     q_t = self._targ_q_func(batch.observations, batch.actions, "min")
    #     v_t = self._value_func(batch.observations)
    #     diff = q_t.detach() - v_t
    #     weight = (self._expectile - (diff < 0.0).float()).abs().detach()
    #     return (weight * (diff**2)).mean()
    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        actions, _ = self._policy.sample_with_log_prob(batch.observations)
        q_t = self._targ_q_func(batch.observations, actions, "min")
        # q_t = q_t.clamp(-1500, 1500)
        v_t = self._value_func(batch.observations)
        # v_t = v_t.clamp(-1500, 1500)
        diff = q_t.detach() - v_t
        return ((diff ** 2)).mean()
    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        v_loss = self.compute_value_loss(batch) - self.compute_conservative_value_loss(batch)[0]

        self._critic_optim2.zero_grad()
        v_loss.backward()
        self._critic_optim2.step()

        q_tpn = self.compute_target(batch)
        # q_tpn = q_tpn.clamp(-1500, 1500)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        self._critic_optim1.zero_grad()
        q_loss.backward()
        self._critic_optim1.step()

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()

    # def compute_actor_loss(self,batch: TorchMiniBatch) -> torch.Tensor:
    #     assert self._policy is not None
    #     dist = self._policy.dist(batch.observations)
    #     log_probs = dist.log_prob(batch.actions)
    #     action, log_probs2 = self._policy.sample_with_log_prob(batch.observations)
    #     action = action.detach()
    #     log_probs2 = dist.log_prob(action)
    #     # compue weight
    #     with torch.no_grad():
    #         weight = self._compute_weight(batch)
    #         baselines = self._value_func(batch.observations )
    #         observation_next, reward = self.dynamics.predict_gpu(batch.observations, action, False)
    #         advantages2 = self._value_func(observation_next ) + reward - baselines
    #         weight2=(self._weight_temp * advantages2).exp().clamp(max=self._max_weight)
    #
    #
    #     return -self._bc*(weight * log_probs).mean()-self._expectile*(weight2 * log_probs2).mean()



    def compute_actor_loss1(self,batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        # action = action.detach()
        # log_probs2 = dist.log_prob(action)
        # compue weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return -self._bc*(weight * log_probs).mean()



    def compute_actor_loss2(self,batch: TorchMiniBatch) -> torch.Tensor:
        action, log_probs2 = self._policy.sample_with_log_prob(batch.observations)
        observation_next, reward = self.dynamics.predict_gpu2(batch.observations, action, False)
        # advantages2 = self._gamma*self._value_func(observation_next ) + reward
        advantages2 = self._value_func(observation_next)
        advantages2 = advantages2.clamp(-1500, 1500)
        # lam = self._expectile * 100/ (advantages2.abs().mean()).detach()
        return -self._expectile*(advantages2).mean(),(advantages2.mean()).detach()
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        loss1 = self.compute_actor_loss1(batch)
        loss2,adv2 = self.compute_actor_loss2(batch)
        return loss1,loss2,adv2
