import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM, RandomRLM
import os
import argparse

TASKLIST_TRAIN = [
    "tomato salad", "lettuce salad", "onion salad",
    "lettuce-tomato salad", "onion-tomato salad",
    "lettuce-onion salad", "lettuce-onion-tomato salad"
]
MAPTYPES_TRAIN = ["A", "B", "C"]

def define_env(env_config_for_registration):
    default_reward_config = {
        "metatask failed": 0, "goodtask finished": 5, "subtask finished": 10,
        "correct delivery": 200, "wrong delivery": -50, "step penalty": -1.,
    }
    default_env_params = {
        "grid_dim": env_config_for_registration.get("grid_dim", [5, 5]),
        "task": env_config_for_registration.get("task", "tomato salad"),
        "rewardList": default_reward_config,
        "map_type": env_config_for_registration.get("map_type", "A"),
        "mode": "vector",
        "debug": False,
        "human_role": env_config_for_registration.get("human_role", "helpful"),
        "frame_stack_size": env_config_for_registration.get("frame_stack_size", 4)
    }
    register_env(
        "Overcooked",
        lambda config_dict: Overcooked_multi(**{**default_env_params, **config_dict}),
    )
    return default_env_params

def define_agents(args):
    '''
    Define the human agent policy and the policies to train.
    Can easily be extended to also define the AI policy
    :param args:
    :return: RLModuleSpec for the human agent, list for policies to train
    '''
    human_policy = RLModuleSpec()
    policies_to_train = ['ai']
    #if args.rl_module == 'stationary':
    #    human_policy = RLModuleSpec(module_class=AlwaysStationaryRLM)
    #    policies_to_train = ['ai']
    #elif args.rl_module == 'random':
    #    human_policy = RLModuleSpec(module_class=RandomRLM)
    #    policies_to_train = ['ai']
    #elif args.rl_module == 'learned':
    #    human_policy = RLModuleSpec()
    #    policies_to_train = ['ai', 'human']
    #else:
    #    raise NotImplementedError(f"{args.rl_module} not a valid human agent")
    return human_policy, policies_to_train

def define_training(human_policy, policies_to_train):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Overcooked")
        .env_runners(
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0
        )
        .multi_agent(
            policies={"ai", "human"},
            policy_mapping_fn=lambda aid, *a, **kw: aid,
            policies_to_train=policies_to_train

        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "human": human_policy,
                    "ai": RLModuleSpec(),
                }
            ),
        )
        .training(
            lr=1e-3,
            lambda_=0.98,
            gamma=0.99,
            clip_param=0.05,
            entropy_coeff=0.1,
            vf_loss_coeff=0.1,
            grad_clip=0.1,
            num_epochs=10,
            minibatch_size=64,
        )
    )
    return config


def train(args, config, env_params):
    """Initializes Ray and runs the training tuner."""
    ray.init(ignore_reinit_error=True)
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    experiment_name = (
        f"{args.name}_task-{env_params['task'].replace(' ', '_')}_map-{env_params['map_type']}_"
        f"role-{env_params['human_role']}_fs-{env_params['frame_stack_size']}_{int(time.time() * 1000)}"
    )
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop={"training_iteration": args.stop_iters},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=2
            ),
        )
    )
    tuner.fit()

def main(args):
    """Main function to set up and run training."""
    env_config = {
         "grid_dim": args.grid_dim,
         "task": args.task,
         "map_type": args.map_type,
         "human_role": args.human_role,
         "frame_stack_size": args.frame_stack_size,
         "mode": "vector",
    }
    final_env_params = define_env(env_config)

    human_policy, policies_to_train = define_agents(args)
    config = define_training(human_policy, policies_to_train)
    config = config.environment("Overcooked", env_config=env_config)

    train(args, config, final_env_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument(
        "--task", default="tomato salad", type=str,
        choices=TASKLIST_TRAIN,
        help="Recipe/task for the environment."
    )
    parser.add_argument(
        "--map_type", default="A", type=str,
        choices=MAPTYPES_TRAIN,
        help="Map layout type."
    )

    parser.add_argument(
        "--human_role", default="helpful", type=str,
        choices=["helpful", "adversarial", "random"],
        help="Set the fixed role for the human agent."
    )
    parser.add_argument(
        "--frame_stack_size", default=4, type=int,
        help="Number of frames to stack for observations."
    )
    parser.add_argument(
        "--stop_iters", default=200, type=int,
        help="Number of training iterations to run."
    )
    parser.add_argument(
        "--checkpoint_freq", default=50, type=int,
        help="Frequency of saving checkpoints (in iterations)."
    )
    parser.add_argument(
        '--grid_dim', type=int, nargs=2, default=[9, 9],
        help='Grid world dimensions (e.g., 9 9).'
    )

    args = parser.parse_args()
    if args.task not in TASKLIST_TRAIN:
        raise ValueError(f"Invalid task provided: {args.task}")
    if args.map_type not in MAPTYPES_TRAIN:
         raise ValueError(f"Invalid map_type provided: {args.map_type}")

    main(args)