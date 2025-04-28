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
        "correct delivery": 200, "wrong delivery": -50, "step penalty": 0.0,
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
    policies_to_train=['ai', 'human']
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

def define_training(human_policy, policies_to_train, learning_rate):
    """Defines the PPO training configuration."""
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
             "Overcooked",
        )
        .env_runners(
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0
        )
        .multi_agent(
            policies={"ai", "human"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
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
            lr=learning_rate,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            num_epochs=10,
            minibatch_size=128,
            train_batch_size=2000,
            vf_clip_param=10.0,
            model={"fcnet_hiddens": [128, 128]}
        )
    )
    return config

def get_curriculum_task(level):
    if level == 1:
        return "tomato salad"  # Simplest
    elif level == 2:
        return "lettuce-tomato salad"  # Medium
    else:
        return "lettuce-onion-tomato salad"  # Complex


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
    """Sets up environment, agents, config, and runs training."""
    env_config = {
         "grid_dim": args.grid_dim,
         "task": args.task,
         "map_type": args.map_type,
         "human_role": args.human_role,
         "frame_stack_size": args.frame_stack_size,
         "mode": "vector",
         "rewardList": {
            "metatask failed": -5.0,
            "subtask finished": 10.0,
            "correct delivery": 200.0,
            "wrong delivery": -50.0,
            "step penalty": -0.01,
          }
    }

    register_env(
        "Overcooked",
        lambda config_dict: Overcooked_multi(**config_dict),
    )

    human_policy, policies_to_train = define_agents(args)

    config_obj = define_training(human_policy, policies_to_train, args.lr)
    config_obj = config_obj.environment("Overcooked", env_config=env_config)

    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    experiment_name = (
        f"{args.name}_task-{env_config['task'].replace(' ', '_')}_map-{env_config['map_type']}_"
        f"grid-{env_config['grid_dim'][0]}x{env_config['grid_dim'][1]}_role-{env_config['human_role']}_"
        f"fs-{env_config['frame_stack_size']}_lr-{args.lr}_{int(time.time() * 1000)}"
    )

    if not ray.is_initialized():
         ray.init(ignore_reinit_error=True)

    if args.task == "auto":
        env_config["task"] = get_curriculum_task(args.curriculum_level)
    else:
        env_config["task"] = args.task

    tuner = tune.Tuner(
        "PPO",
        param_space=config_obj.to_dict(),
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop={"training_iteration": args.stop_iters},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=3
            ),
        )
    )

    results = tuner.fit()

    print("Training finished.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    if best_result and best_result.checkpoint:
        print(f"Best checkpoint found at: {best_result.checkpoint.path}")
        print(f"With episode_reward_mean: {best_result.metrics.get('episode_reward_mean', 'N/A')}")
    else:
        print("Could not determine best checkpoint.")

    ray.shutdown()

if __name__ == "__main__":
    import argparse # Ensure imported

    TASKLIST_TRAIN = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    MAPTYPES_TRAIN = ["A", "B", "C"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str, help="Directory to save training results.")
    parser.add_argument("--name", default="run", type=str, help="Base name for the experiment directory.")
    parser.add_argument("--task", default="tomato salad", type=str, choices=TASKLIST_TRAIN, help="Recipe/task name.")
    parser.add_argument("--map_type", default="A", type=str, choices=MAPTYPES_TRAIN, help="Map layout type.")
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help="Grid dimensions (e.g., 5 5).")
    parser.add_argument(
        "--human_role",
        default="random_h_n_a", # Default to the new random mode
        type=str,
        choices=["helpful", "adversarial", "neutral", "random_h_a", "random_h_n_a"],
        help="Set the human partner's role/incentive structure."
    )
    parser.add_argument("--frame_stack_size", default=4, type=int, help="Number of frames to stack.")
    parser.add_argument("--stop_iters", default=300, type=int, help="Number of training iterations.") # Increased default
    parser.add_argument("--checkpoint_freq", default=50, type=int, help="Checkpoint frequency (iterations).")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for PPO.") # Default 1e-4
    parser.add_argument("--rl_module", default="stationary", choices=["stationary", "random", "learned"], help="Set the policy type of the human agent during training.")
    parser.add_argument("--curriculum_level", default=1, type=int, 
                   help="Difficulty level: 1=single ingredient, 2=two ingredients, 3=three ingredients")

    args = parser.parse_args()
    

    if args.task not in TASKLIST_TRAIN: raise ValueError(f"Invalid task: {args.task}")
    if args.map_type not in MAPTYPES_TRAIN: raise ValueError(f"Invalid map_type: {args.map_type}")
    if args.frame_stack_size <= 0: raise ValueError("frame_stack_size must be positive.")
    if args.stop_iters <= 0: raise ValueError("stop_iters must be positive.")

    main(args)
