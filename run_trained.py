import glob
import time
from environment.Overcooked import Overcooked_multi
from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.core.columns import Columns
import torch
import os
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np
from Agents import (
    AlwaysStationaryRLM, 
    RandomRLM,
    ObstructingRLM,
    ClumsyRLM,
    SelfishRLM,
    HelpfulRLM
)
from ray.rllib.algorithms.algorithm import Algorithm

#reward_config = {
#    "metatask failed": 0,
#    "goodtask finished": 5,
#    "subtask finished": 10,
#    "correct delivery": 200,
#    "wrong delivery": -50,
#    "step penalty": -1.,
#}
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

env = Overcooked_multi(**env_params)


def sample_action(mdl, obs):
    mdl_out = mdl.forward_inference({Columns.OBS: obs})
    if Columns.ACTION_DIST_INPUTS in mdl_out: #our custom policies might return the actions directly, while learned policies might return logits.
        logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])
        action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
        return action
    elif 'actions' in mdl_out:
        return mdl_out['actions'][0]

    else:
        raise NotImplementedError("Something weird is going on when sampling acitons")

def load_modules(args):
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    p = f"{storage_path}/{args.name}_{args.rl_module}_*"
    experiment_name = glob.glob(p)[-1]
    print(f"Loading results from {experiment_name}...")
    restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint
    human_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'human',
    ))
    ai_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'ai',
    ))
    return ai_module, human_module

def get_human_agent_from_config(config):
    """Extract the human agent type from the config dictionary"""
    if "_rl_module_spec" in config:
        spec = config["_rl_module_spec"]
        # Look for module_class in the human agent specification
        if hasattr(spec, "rl_module_specs") and "human" in spec.rl_module_specs:
            human_spec = spec.rl_module_specs["human"]
            if hasattr(human_spec, "module_class"):
                module_class = human_spec.module_class
                if module_class == ObstructingRLM:
                    return "obstruct"
                elif module_class == ClumsyRLM:
                    return "clumsy"  
                elif module_class == SelfishRLM:
                    return "selfish"
                elif module_class == HelpfulRLM:
                    return "helpful"
                elif module_class == RandomRLM:
                    return "random"
    # Default or stationary
    return "stationary"


def main(args):
    print(f"Loading from {args.checkpoint_path}...")
    
    # Check if the path is a direct checkpoint or a directory
    if os.path.isfile(args.checkpoint_path) or (os.path.isdir(args.checkpoint_path) and "checkpoint_" in args.checkpoint_path):
        # Direct path to a checkpoint was provided
        latest_checkpoint = args.checkpoint_path
        print(f"Using direct checkpoint: {latest_checkpoint}")
    else:
        # Directory path was provided - search for checkpoints
        checkpoint_path = glob.glob(os.path.join(args.checkpoint_path, "**", "checkpoint_*"), recursive=True)
        checkpoint_path.sort()  # Sort to get the latest
        
        if not checkpoint_path:
            print(f"No checkpoints found in {args.checkpoint_path}")
            return
            
        latest_checkpoint = checkpoint_path[-1]
        print(f"Found checkpoint: {latest_checkpoint}")
    
    # Load the algorithm config
    try:
        algo_config = Algorithm.from_checkpoint(latest_checkpoint).get_config()
        print("Config loaded successfully")
        
        # Get the human agent type from the config
        human_agent_type = get_human_agent_from_config(algo_config)
        print(f"Detected human agent type: {human_agent_type}")
        
        # Override with command line argument if provided
        if args.human_agent:
            human_agent_type = args.human_agent
            print(f"Overriding with human agent type from command line: {human_agent_type}")
        
        # Create human module based on type
        if human_agent_type == "obstruct":
            human_module = ObstructingRLM()
        elif human_agent_type == "clumsy":
            human_module = ClumsyRLM()
        elif human_agent_type == "selfish":
            human_module = SelfishRLM()
        elif human_agent_type == "helpful":
            human_module = HelpfulRLM()
        elif human_agent_type == "random":
            human_module = RandomRLM()
        else:
            human_module = AlwaysStationaryRLM()
            
        # Load AI module from checkpoint
        print("Loading AI module from checkpoint...")
        try:
            ai_module = RLModule.from_checkpoint(os.path.join(
                latest_checkpoint,
                COMPONENT_LEARNER_GROUP,
                COMPONENT_LEARNER,
                COMPONENT_RL_MODULE,
                'ai',
            ))
            print("AI module loaded successfully")
        except Exception as e:
            print(f"Error loading AI module: {e}")
            return
            
        # Run environment
        env.game.on_init()
        obs, info = env.reset()
        env.render()

        while True:
            ai_action = sample_action(ai_module, torch.from_numpy(obs['ai']).unsqueeze(0).float())
            human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float())
            actions = {'human': human_action, 'ai': ai_action}

            obs, rewards, terminateds, _, _ = env.step(actions)
            env.render()
            time.sleep(0.1)

            if terminateds['__all__']:
                break
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the checkpoint directory (e.g., runs/run_obstruct_1748810782133)")
    parser.add_argument("--human_agent", default=None, choices=[None, "stationary", "random", "obstruct", "clumsy", "selfish", "helpful"],
                      help="Override the human agent type (default: use the one from training)")
    
    args = parser.parse_args()
    main(args)
