import argparse
import d3rlpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int,default=0)
    parser.add_argument('--expectile', type=float, default=0)
    parser.add_argument('--weight_temp', type=float, default=30)
    parser.add_argument('--conservative_weight', type=float, default=1)
    parser.add_argument('--bc', type=float, default=1)
    parser.add_argument('--alphalr', type=float, default=1e-4)
    parser.add_argument('--alphath', type=float, default=-10)
    parser.add_argument('--adroit', type=int, default=1)
    args = parser.parse_args()
    if args.adroit == 0:
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    else:
        dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # PATH='../offline/d3rlpy_logs/dynamics_expert_5.pt'
    # PATH_json = '../offline/d3rlpy_logs/dynamics_expert_5.json'

    PATH = f'../../../model/model_process_v2_5/dynamics_{args.dataset}.pt'
    PATH_json = f'../../../model/params_process_v2_5/dynamics_{args.dataset}.json'
    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)
    dynamics =d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(PATH_json)
    dynamics.load_model(PATH)
    # uncomment this when dealing with dataset with very different reward range, e.g. hammer-expert
    # reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
    #    multiplier=5000.0)

    csve= d3rlpy.algos.CSVE(actor_learning_rate=3e-3,
                           critic_learning_rate=3e-3,
                           batch_size=256,
                           weight_temp=args.weight_temp,
                           max_weight=1000.0,
                           expectile=args.expectile,
                           alpha_learning_rate=args.alphalr,
                           alpha_threshold=args.alphath,
                           #reward_scaler=reward_scaler,
                           use_gpu=args.gpu,dynamics=dynamics,conservative_weight=args.conservative_weight,bc=args.bc)

    # # workaround for learning scheduler
    csve.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
    scheduler = CosineAnnealingLR(csve.impl._actor_optim, 1000000)

    def callback(algo, epoch, total_step):
        scheduler.step()

    csve.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=100000,
            n_steps_per_epoch=100,
            save_interval=10,
            callback=callback,
            tensorboard_dir='exp_09',
            logdir='logs_09',
            scorers={
                'environment': d3rlpy.metrics2.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"0922_Final_csve_{args.dataset}_exp=={args.expectile}_temp=={args.weight_temp}_alphath=={args.alphath}")


if __name__ == '__main__':
    main()
