import glob
import time
import os
import argparse
import torch
import numpy as np
from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP, # Default: "learner_group_0"
    COMPONENT_LEARNER,       # Default: "learner_0"
    COMPONENT_RL_MODULE      # Default: "module_0" or "rl_module" -> Needed for path
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,      # Default: "env_runners"
    EPISODE_RETURN_MEAN,     # Default: "episode_return_mean"
)
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import traceback

try:
    # Adjust path if your environment module is elsewhere
    from environment.Overcooked import Overcooked_multi, TASKLIST as TASKLIST_ENV, MAPTYPES_TRAIN
except ImportError:
    print("Error: Could not import Overcooked environment. Check path.")
    TASKLIST_ENV = ["tomato salad"]
    MAPTYPES_TRAIN = ["A", "B", "C"]

def sample_action(mdl, obs):
    """
    Samples an action from the RLModule given an observation.
    Handles None modules and different output formats.
    """
    if mdl is None:
        return 4

    if not isinstance(obs, torch.Tensor):
        if obs is None or obs.size == 0:
             print("Warning: Received None or empty observation in sample_action. Returning default action 4.")
             return 4
        try:
            obs = torch.from_numpy(np.asarray(obs)).float()
        except TypeError as e:
             print(f"Error converting observation to tensor: {e}. Obs type: {type(obs)}. Returning default action 4.")
             return 4

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    elif obs.ndim == 0:
         print("Warning: Received scalar observation. Unsqueezing twice.")
         obs = obs.unsqueeze(0).unsqueeze(0)
    elif obs.ndim > 2:
         print(f"Warning: Observation has unexpected dimensions {obs.ndim}. Using first element.")
         obs = obs[0]
         if obs.ndim == 1: obs = obs.unsqueeze(0)

    try:
        with torch.no_grad():
            mdl_out = mdl.forward_inference({Columns.OBS: obs})
    except Exception as e_inf:
        print(f"Error during RLModule forward_inference: {e_inf}. Returning default action 4.")
        traceback.print_exc()
        return 4

    if Columns.ACTION_DIST_INPUTS in mdl_out:
        logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])
        if logits is not None and logits.ndim > 1 and logits.shape[0] > 0 and logits.shape[1] > 0:
            try:
                probabilities = softmax(logits[0])
                if np.isnan(probabilities).any() or not np.isclose(np.sum(probabilities), 1.0):
                     print(f"Warning: Invalid probabilities calculated {probabilities}. Sampling uniformly.")
                     num_actions = len(probabilities)
                     action = np.random.choice(list(range(num_actions)))
                else:
                     action = np.random.choice(list(range(len(probabilities))), p=probabilities)
                return action
            except Exception as e_sample:
                 print(f"Error sampling action from distribution: {e_sample}. Sampling uniformly.")
                 num_actions = logits.shape[1]
                 return np.random.choice(list(range(num_actions)))
        else:
             print(f"Warning: Unexpected logits shape {logits.shape if logits is not None else 'None'}. Sampling uniformly.")
             num_actions = getattr(getattr(mdl, 'action_space', None), 'n', 5)
             return np.random.choice(list(range(num_actions)))
    elif Columns.ACTIONS in mdl_out:
        actions_out = convert_to_numpy(mdl_out[Columns.ACTIONS])
        if actions_out is not None and actions_out.size > 0:
             action = actions_out.item(0)
             return int(action)
        else:
             print(f"Warning: Actions output is None or empty {actions_out}. Returning default action 4.")
             return 4
    else:
        raise ValueError("Could not find action outputs (ACTION_DIST_INPUTS or ACTIONS) in RLModule inference result.")


def load_modules(experiment_path):
    """Loads AI and Human RL Modules from the specified experiment path."""
    print(f"Attempting to load results from: {experiment_path}")
    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

    best_checkpoint = None
    # Try restoring tuner to find best checkpoint by metric first
    try:
        print("Attempting to restore Tuner to find best checkpoint by metric...")
        restored_tuner = tune.Tuner.restore(experiment_path, trainable="PPO")
        result_grid = restored_tuner.get_results()
        possible_metric_keys = [
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
            "episode_reward_mean",
            "sampler_results/episode_reward_mean"
        ]
        best_result = None
        for metric_key in possible_metric_keys:
             if len(result_grid) > 0:
                 try:
                     print(f"Trying metric key: {metric_key}")
                     best_result = result_grid.get_best_result(metric=metric_key, mode="max")
                     if best_result and best_result.checkpoint:
                          print(f"Found best result via metric '{metric_key}'")
                          best_checkpoint = best_result.checkpoint
                          print(f"Using checkpoint from best result: {best_checkpoint.path}")
                          break
                     else: print(f"Metric '{metric_key}' found but no valid checkpoint.")
                 except Exception as e_metric: print(f"Metric '{metric_key}' not found or error: {e_metric}")
             else: print("Result grid empty."); break
        if best_checkpoint is None: print("Could not find best checkpoint via metrics. Trying fallback.")
    except Exception as e: print(f"Could not restore Tuner or find best result by metric: {e}. Trying fallback.")

    # Fallback: Find the latest checkpoint directory manually
    if best_checkpoint is None:
        print("Finding latest checkpoint directory manually...")
        try:
            checkpoint_dirs = glob.glob(os.path.join(experiment_path, "*", "checkpoint_*"))
            if not checkpoint_dirs: checkpoint_dirs = glob.glob(os.path.join(experiment_path, "checkpoint_*"))
            if not checkpoint_dirs: raise ValueError("No checkpoint directories found.")
            latest_checkpoint_dir = max(checkpoint_dirs, key=lambda d: int(d.split('_')[-1]))
            print(f"Using latest checkpoint directory found: {latest_checkpoint_dir}")
            best_checkpoint = type('obj', (object,), {'path': latest_checkpoint_dir})()
        except Exception as e_fallback: print(f"Error finding latest checkpoint: {e_fallback}"); raise

    # --- Load RL Modules ---
    if not best_checkpoint or not hasattr(best_checkpoint, 'path') or not best_checkpoint.path:
         raise ValueError("Failed to determine a valid checkpoint path.")

    # Construct the path including COMPONENT_RL_MODULE
    module_base_path = os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP, # Default: "learner_group_0"
        COMPONENT_LEARNER,       # Default: "learner_0"
        COMPONENT_RL_MODULE      # Default: "module_0" or "rl_module"
    )
    print(f"Constructed module base path (including COMPONENT_RL_MODULE): {module_base_path}")

    # Check if this base path exists before proceeding
    if not os.path.isdir(module_base_path):
         print(f"Warning: Path with COMPONENT_RL_MODULE not found: {module_base_path}. Trying path without it.")
         module_base_path = os.path.join(
             best_checkpoint.path,
             COMPONENT_LEARNER_GROUP,
             COMPONENT_LEARNER
         )
         print(f"Trying fallback module base path: {module_base_path}")
         if not os.path.isdir(module_base_path):
              raise FileNotFoundError(f"Could not find learner directory structure inside checkpoint: {best_checkpoint.path}")


    # Append policy IDs to the determined base path
    ai_module_path = os.path.join(module_base_path, 'ai')
    human_module_path = os.path.join(module_base_path, 'human')


    ai_module = None
    human_module = None

    # Load AI module
    if os.path.exists(ai_module_path):
        print(f"Loading AI module from: {ai_module_path}")
        try:
            ai_module = RLModule.from_checkpoint(ai_module_path)
            print("AI module loaded successfully.")
        except Exception as e_ai:
            print(f"Error loading AI module: {e_ai}")
            traceback.print_exc()
            raise ValueError(f"Failed to load essential AI module: {e_ai}")
    else:
        print(f"AI module directory not found at: {ai_module_path}")
        alt_base_path = os.path.join(best_checkpoint.path, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER)
        if module_base_path != alt_base_path:
             print(f"Also checked fallback path: {os.path.join(alt_base_path, 'ai')}")
        raise FileNotFoundError("Essential AI module not found in checkpoint.")

    if os.path.exists(human_module_path):
        print(f"Loading Human module from: {human_module_path}")
        try:
            human_module = RLModule.from_checkpoint(human_module_path)
            print("Human module loaded successfully.")
        except Exception as e_human:
            print(f"Warning: Error loading Human module (might be expected): {e_human}")
            human_module = None
    else:
        print(f"Human module directory not found at: {human_module_path} (expected if not trained).")

    return ai_module, human_module


def main(args):
    """Loads trained modules and runs evaluation episodes."""
    reward_config = {
        "metatask failed": 0, "goodtask finished": 5, "subtask finished": 10,
        "correct delivery": 200, "wrong delivery": -50, "step penalty": -1.,
    }
    env_params = {
        "grid_dim": args.grid_dim,
        "task": args.task,
        "rewardList": reward_config,
        "map_type": args.map_type,
        "mode": "vector",
        "debug": args.debug,
        "frame_stack_size": args.frame_stack_size,
        "human_role": args.human_role
    }
    print(f"Creating evaluation environment with parameters: {env_params}")
    try:
        env = Overcooked_multi(**env_params)
    except Exception as e_env:
        print(f"Error creating environment: {e_env}")
        return

    render_possible = False
    if hasattr(env, 'game') and env.game:
         try:
             env.game.on_init()
             render_possible = True
         except Exception as e_render:
             print(f"Warning: Pygame initialization failed: {e_render}. Rendering will be disabled.")
    else:
         print("Warning: Environment does not have 'game' attribute for rendering initialization. Rendering disabled.")

    try:
        current_dir = os.getcwd()
        if not args.experiment_path:
             storage_path = os.path.join(current_dir, args.save_dir)
             pattern = os.path.join(storage_path, f"{args.name}*")
             print(f"Searching for latest experiment matching pattern: {pattern}")
             experiment_folders = glob.glob(pattern)
             if not experiment_folders:
                  print(f"Error: No experiment found matching name prefix '{args.name}' in '{storage_path}'")
                  return
             experiment_path = max(experiment_folders, key=os.path.getmtime)
             print(f"Found latest experiment: {experiment_path}")
        else:
             experiment_path = args.experiment_path

        ai_module, human_module = load_modules(experiment_path)
    except (FileNotFoundError, ValueError) as e_load:
         print(f"Error loading modules: {e_load}")
         return

    try:
        num_episodes = args.num_episodes
        all_ai_rewards = []
        all_human_rewards = []
        all_steps = []

        print(f"\nRunning {num_episodes} evaluation episode(s)...")

        for ep in range(num_episodes):
            print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")
            try:
                obs, info = env.reset()
                if render_possible: env.render()
                print(f"Episode Start. Human role: {env.human_role} (Multiplier: {env.human_multiplier})")
            except Exception as e_reset:
                print(f"Error during environment reset: {e_reset}. Skipping episode.")
                traceback.print_exc()
                continue

            episode_reward_ai = 0
            episode_reward_human = 0
            steps = 0
            max_steps = args.max_steps

            done = False
            truncated = False
            while not done and not truncated and steps < max_steps:
                try:
                    ai_obs_np = np.asarray(obs.get('ai')) if obs.get('ai') is not None else None
                    human_obs_np = np.asarray(obs.get('human')) if obs.get('human') is not None else None

                    ai_action = sample_action(ai_module, ai_obs_np)
                    human_action = sample_action(human_module, human_obs_np)

                    actions = {'human': human_action, 'ai': ai_action}

                    next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
                    obs = next_obs 

                    if render_possible: env.render()
                    time.sleep(args.sleep)

                    episode_reward_ai += rewards.get('ai', 0)
                    episode_reward_human += rewards.get('human', 0)
                    steps += 1

                    done = terminateds.get('__all__', False)
                    truncated = truncateds.get('__all__', False)

                except Exception as e_step:
                    print(f"\nError during environment step {steps+1}: {e_step}")
                    traceback.print_exc()
                    break

            print(f"Episode {ep + 1} finished after {steps} steps.")
            print(f"Episode Rewards: AI={episode_reward_ai:.2f}, Human={episode_reward_human:.2f}")
            all_ai_rewards.append(episode_reward_ai)
            all_human_rewards.append(episode_reward_human)
            all_steps.append(steps)

        print("")
        print("--- Evaluation Summary ---")
        print(f"Ran {num_episodes} episodes.")
        if all_steps:
             print(f"Avg Steps per Episode: {np.mean(all_steps):.2f} (+/- {np.std(all_steps):.2f})")
        if all_ai_rewards:
             print(f"Avg AI Reward: {np.mean(all_ai_rewards):.2f} (+/- {np.std(all_ai_rewards):.2f})")
        if all_human_rewards:
             print(f"Avg Human Reward: {np.mean(all_human_rewards):.2f} (+/- {np.std(all_human_rewards):.2f})")


    except Exception as e:
        print(f"\nAn error occurred during evaluation loop: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
             try:
                 env.close()
             except Exception as e_close:
                 print(f"Error closing environment: {e_close}")
        print("Evaluation run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Overcooked agents.")
    parser.add_argument("--save_dir", default="runs", type=str, help="Base directory where training runs are stored.")
    parser.add_argument("--name", default="run", type=str, help="Prefix of the experiment name (used to find latest if --experiment_path is not set).")
    parser.add_argument("--experiment_path", default=None, type=str, help="Full path to the specific experiment directory to load checkpoint from.")
    parser.add_argument(
        "--task", default="tomato salad", type=str, choices=TASKLIST_ENV,
        help="Recipe/task for the evaluation environment."
    )
    parser.add_argument(
        "--map_type", default="A", type=str, choices=MAPTYPES_TRAIN,
        help="Map layout type for evaluation."
    )
    parser.add_argument(
        "--human_role", default="helpful", type=str, choices=["helpful", "adversarial", "random"],
        help="Set the human role for this evaluation run."
    )
    parser.add_argument(
        "--frame_stack_size", default=4, type=int,
        help="Number of frames stacked (MUST match training configuration)."
    )
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help='Grid world size for evaluation env.')
    parser.add_argument("--sleep", default=0.1, type=float, help="Time delay (seconds) between steps for rendering.")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints in the environment.")
    parser.add_argument("--num_episodes", default=3, type=int, help="Number of evaluation episodes to run.") # Increased default
    parser.add_argument("--max_steps", default=200, type=int, help="Maximum steps per evaluation episode.")

    args = parser.parse_args()

    if args.task not in TASKLIST_ENV: raise ValueError(f"Invalid task: {args.task}")
    if args.map_type not in MAPTYPES_TRAIN: raise ValueError(f"Invalid map_type: {args.map_type}")
    if args.frame_stack_size <= 0: raise ValueError("frame_stack_size must be positive.")
    if args.num_episodes <= 0: raise ValueError("num_episodes must be positive.")

    main(args)
