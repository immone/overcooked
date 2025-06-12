import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import (
    AlwaysStationaryRLM, 
    RandomRLM,
    ObstructingRLM,
    ClumsyRLM,
    SelfishRLM,
    HelpfulRLM
)
import os
import matplotlib.pyplot as plt


def define_env():
    reward_config = {
        "metatask failed": -10,  # Increased penalty for metatask failure to discourage it
        "goodtask finished": 20,  # Increased reward for good task completion to encourage it
        "subtask finished": 50,  # Increased reward for subtask completion to encourage it
        "correct delivery": 800,  # Increased reward for correct delivery to encourage it
        "wrong delivery": -100,  # Increased penalty for wrong delivery to discourage it
        "step penalty": -0.3,  # Reduced step penalty to encourage exploration
        # New rewards
        "correct_ingredient_pickup": 15,
        "sequence_progress": 10,
        "sequence_completion": 50,
    }
    env_params = {
        "grid_dim": [5, 5],  # Reduced grid size for faster training
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "mode": "vector",
        "debug": False,
    }

    register_env(
        "Overcooked",
        lambda _: Overcooked_multi(**env_params),
    )

def define_agents(args):
    '''
    Define the human agent policy and the policies to train.
    Can easily be extended to also define the AI policy
    :param args:
    :return: RLModuleSpec for the human agent, list for policies to train
    '''
    if args.rl_module == 'stationary':
        human_policy = RLModuleSpec(module_class=AlwaysStationaryRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'random':
        human_policy = RLModuleSpec(module_class=RandomRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'learned':
        human_policy = RLModuleSpec()
        policies_to_train = ['ai', 'human']
    elif args.rl_module == 'obstruct':
        human_policy = RLModuleSpec(module_class=ObstructingRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'clumsy':
        human_policy = RLModuleSpec(module_class=ClumsyRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'selfish':
        human_policy = RLModuleSpec(module_class=SelfishRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'helpful':
        human_policy = RLModuleSpec(module_class=HelpfulRLM)
        policies_to_train = ['ai']
    else:
        raise NotImplementedError(f"{args.rl_module} not a valid human agent")
        
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
            num_env_runners=4,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=2,
            num_gpus_per_env_runner=0.25,
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
            lr=5e-4,
            lambda_=0.98,
            gamma=0.99,
            clip_param=0.05,
            entropy_coeff=0.2,
            vf_loss_coeff=0.1,
            grad_clip=0.1,
            num_epochs=10,
            minibatch_size=64,
        )
        # Remove this line:
        # .callbacks(OvecookedMetricsCallback)
    )
    return config


def train(args, config):
    ray.init(include_dashboard=True, ignore_reinit_error=True) #local_mode=True)
    #ray.init()
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir) # save the results in the runs folder
    experiment_name = f"{args.name}_{args.rl_module}_{int(time.time() * 1000)}" # add a timestamp to the name to make it unique
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop={"training_iteration": 200},  # Increase from 200 to 500 or more
            checkpoint_config=CheckpointConfig(checkpoint_frequency=5, checkpoint_at_end=True, num_to_keep=5),
        )
    )
    tuner.fit()
    
    results = tuner.get_results()
    trial_result = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
    rewards = trial_result.metrics_dataframe["env_runners/episode_return_mean"]
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Reward")
    plt.savefig("training_progress.png")
    
    # Remove or comment out the custom metrics plotting:
    '''
    # Plot custom metrics
    metrics = ["custom_metrics/task_completed", "custom_metrics/steps_to_complete", 
               "custom_metrics/final_reward", "custom_metrics/cooperation_score"]

    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        if metric in trial_result.metrics_dataframe:
            plt.plot(trial_result.metrics_dataframe[metric])
            plt.title(metric.split('/')[-1].replace('_', ' ').title())
            plt.xlabel("Iterations")
            plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("performance_metrics.png")
    '''
    
    # Print TensorBoard command for convenience
    print(f"\nTo monitor training live, run in another terminal:")
    print(f"tensorboard --logdir={storage_path}\n")

def main(args):
    define_env()
    human_policy, policies_to_train = define_agents(args)
    config = define_training(human_policy, policies_to_train)
    train(args, config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", help="Specifies the directory to save results to")
    parser.add_argument("--name", default="run", help="Sets the run name")
    parser.add_argument("--rl_module", default="stationary", 
                       help="Set the policy of the human: stationary, random, learned, obstruct, clumsy, selfish, helpful")
    args = parser.parse_args()
    ip = main(args)