import argparse

from d3rlpy.metrics import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics import dynamics_reward_prediction_error_scorer
import d3rlpy
from sklearn.model_selection import train_test_split

import torch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-random-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--adroit', type=int, default=0)
    args = parser.parse_args()

    # create dataset without masks
    if args.adroit == 0:
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    else:
        dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    print(env)
    _, test_episodes = train_test_split(dataset, test_size=0.2)

    # prepare dynamics model
    dynamics_encoder = d3rlpy.models.encoders.VectorEncoderFactory(
        hidden_units=[200, 200, 200, 200],
        activation='swish',
    )
    dynamics_optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=2.5e-5)
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
        encoder_factory=dynamics_encoder,
        optim_factory=dynamics_optim,
        learning_rate=1e-3,
        n_ensembles=5,
        use_gpu=args.gpu,
    )


    # train dynamics model
    dynamics.fit(dataset.episodes,
                 eval_episodes=test_episodes,
                 n_steps=300000,
                 experiment_name=f"dynamics_{args.dataset}_{args.seed}",
                 scorers={
                     "obs_error": dynamics_observation_prediction_error_scorer,
                     "rew_error": dynamics_reward_prediction_error_scorer,
                 })



if __name__ == '__main__':
    main()
