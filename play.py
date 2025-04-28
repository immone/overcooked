import argparse
import copy
import numpy as np
import pandas as pd
import time
import traceback

try:
    from environment.Overcooked import Overcooked_multi, TASKLIST, MAPTYPES_TRAIN
except ImportError:
    print("Error: Could not import Overcooked environment. Check path.")
    TASKLIST = ["tomato salad"]
    MAPTYPES_TRAIN = ["A"]
    exit()

try:
    from Agents import AlwaysStationaryRLM, RandomRLM
except ImportError:
    print("Error: Could not import Agents. Check path.")
    exit()

try:
    from ray.rllib.core.columns import Columns
except ImportError:
    class Columns: OBS = "obs"; ACTIONS = "actions"
    print("Warning: RLlib Columns not found. Using dummy class.")


class Player:
    # Map keyboard keys to environment action indices
    ACTION_MAPPING = {
        "w": 3, # Up
        "d": 0, # Right
        "a": 2, # Left
        "s": 1, # Down
        "q": 4  # Interact
    }
    REVERSE_ACTION_MAPPING = {v: k for k, v in ACTION_MAPPING.items()}

    # Standard rewards used for playing (can be different from training)
    REWARD_LIST = {
        "subtask finished": 10,
        "metatask failed": -5,
        "correct delivery": 200,
        "wrong delivery": -50, # Increased penalty
        "step penalty": -0.01
    }

    def __init__(self, grid_dim, task_name, map_type, mode, debug, agent_partner_type='human'):
        if task_name not in TASKLIST:
             raise ValueError(f"Invalid task name: {task_name}. Choices: {TASKLIST}")
        if map_type not in MAPTYPES_TRAIN:
             raise ValueError(f"Invalid map type: {map_type}. Choices: {MAPTYPES_TRAIN}")
        valid_partners = ['human', 'stationary', 'random'] # Add 'trained' later
        if agent_partner_type not in valid_partners:
             raise ValueError(f"Invalid agent_partner_type: {agent_partner_type}. Choose from {valid_partners}")

        self.env_params = {
            'grid_dim': grid_dim,
            'task': task_name,
            'rewardList': self.REWARD_LIST,
            'map_type': map_type,
            'mode': mode,
            'debug': debug,
            'frame_stack_size': 1,
            'human_role': 'helpful' 
        }
        print(f"Initializing environment with params: {self.env_params}")
        self.env = Overcooked_multi(**self.env_params)

        self.ai_partner_type = agent_partner_type
        self.ai_partner_agent = None

        if self.ai_partner_type != 'human':
             ai_obs_space = self.env.observation_spaces.get('ai')
             ai_action_space = self.env.action_spaces.get('ai')

             if ai_obs_space is None or ai_action_space is None:
                  raise RuntimeError("Could not retrieve AI observation/action space from environment.")

             print(f"Initializing AI partner: {self.ai_partner_type}")
             if self.ai_partner_type == 'stationary':
                 self.ai_partner_agent = AlwaysStationaryRLM(
                     observation_space=ai_obs_space, action_space=ai_action_space
                 )
             elif self.ai_partner_type == 'random':
                 self.ai_partner_agent = RandomRLM(
                     observation_space=ai_obs_space, action_space=ai_action_space
                 )
             # Placeholder for loading trained agent
             # elif self.ai_partner_type == 'trained':
             #    # Requires loading logic similar to run_trained.py
             #    # checkpoint_path = args.checkpoint_path # Need new arg
             #    # loaded_ai_module, _ = load_modules(checkpoint_path) # Use loading function
             #    # self.ai_partner_agent = loaded_ai_module
             #    print("Trained agent loading not yet implemented in play.py")
             #    pass

        self.total_human_reward = 0
        self.total_ai_reward = 0
        self.step = 0

    def run(self):
        """Runs the interactive game loop."""

        render_possible = False
        if hasattr(self.env, 'game') and self.env.game:
            try:
                self.env.game.on_init()
                render_possible = True
            except Exception as e:
                print(f"Error initializing Pygame: {e}. Rendering might fail.")
        else:
             print("Error: Environment has no 'game' attribute for rendering.")
             return

        try:
            obs_dict, info_dict = self.env.reset()
            if render_possible: self.env.render()
            
            data = [["step", "human_action_key", "ai_action_key", "human_reward", "ai_reward", "done"]]

            while True:
                self.step += 1
                print(f"\n--- Step {self.step} ---")

                human_action_idx = None
                human_action_key = None
                while human_action_idx is None:
                     try:
                          input_key = input("Input Human (w/a/s/d/q) or 'p' to save: ").strip().lower()
                          if input_key == 'p':
                               self.save_data(data)
                               print("Data saved to output_play.csv. Continuing game.")
                               continue 

                          human_action_idx = self.ACTION_MAPPING[input_key]
                          human_action_key = input_key
                     except KeyError:
                          print("Invalid input. Use w, a, s, d, q, or p.")
                     except EOFError:
                          print("EOF detected. Exiting.")
                          raise KeyboardInterrupt

                ai_action_idx = None
                ai_action_key = None
                if self.ai_partner_type == 'human':

                    while ai_action_idx is None:
                         try:
                              input_key_ai = input("Input AI Player (w/a/s/d/q): ").strip().lower()
                              ai_action_idx = self.ACTION_MAPPING[input_key_ai]
                              ai_action_key = input_key_ai
                         except KeyError:
                              print("Invalid input. Use w, a, s, d, q.")
                         except EOFError:
                              print("EOF detected. Exiting.")
                              raise KeyboardInterrupt
                elif self.ai_partner_agent:
                     ai_obs = obs_dict.get('ai')
                     if ai_obs is None:
                          print("Error: AI observation missing from dictionary.")
                          ai_action_idx = 4
                          ai_action_key = str(ai_action_idx)
                     else:
                          if not isinstance(ai_obs, np.ndarray): ai_obs = np.array(ai_obs)
                          batch = {Columns.OBS: ai_obs[np.newaxis, ...]}

                          ai_output = self.ai_partner_agent._forward_inference(batch)
                          action_value = ai_output[Columns.ACTIONS][0]

                          ai_action_idx = int(action_value) if isinstance(action_value, (np.ndarray, np.generic)) else action_value
                          ai_action_key = self.REVERSE_ACTION_MAPPING.get(ai_action_idx, str(ai_action_idx)) # Get key for logging
                     print(f"AI Partner ({self.ai_partner_type}) action: {ai_action_key}")
                else:
                     raise RuntimeError("AI partner agent not correctly initialized or type is invalid.")

                action_to_env = {"human": human_action_idx, "ai": ai_action_idx}
                try:
                    next_obs_dict, rewards, terminateds, truncateds, infos = self.env.step(action_to_env)
                except Exception as e_step:
                     print(f"Error during environment step: {e_step}")
                     traceback.print_exc()
                     break

                obs_dict = next_obs_dict

                # Accumulate and display rewards
                current_human_reward = rewards.get('human', 0)
                current_ai_reward = rewards.get('ai', 0)
                self.total_human_reward += current_human_reward
                self.total_ai_reward += current_ai_reward
                print(f"Rewards: Human={current_human_reward:.2f}, AI={current_ai_reward:.2f} | "
                      f"Total: Human={self.total_human_reward:.2f}, AI={self.total_ai_reward:.2f}")

                row = [
                    self.step,
                    human_action_key,
                    ai_action_key,
                    current_human_reward,
                    current_ai_reward,
                    terminateds.get('__all__', False)
                ]
                data.append(copy.deepcopy(row))

                # Render the new state
                if render_possible: self.env.render()
                time.sleep(0.05) # Small delay for visibility

                # Check for episode end (termination or truncation)
                if terminateds.get('__all__', False) or truncateds.get('__all__', False):
                    print("-- Episode finished ---")
                    print(f"Reason: {'Terminated' if terminateds.get('__all__', False) else 'Truncated'}")
                    print(f"Final Score: Human={self.total_human_reward:.2f}, AI={self.total_ai_reward:.2f}")
                    self.save_data(data) # Save data at end
                    break # Exit the loop

        except KeyboardInterrupt:
            print("Game interrupted by user. Saving data...")
            self.save_data(data)
        except Exception as e:
            print(f"An error occurred during the game: {e}")
            traceback.print_exc()
            self.save_data(data)
        finally:
            if 'env' in locals() and hasattr(env, 'close'):
                 self.env.close()
            print("Game exited.")


    def save_data(self, data):
        """Saves collected step data to a CSV file."""
        if len(data) <= 1: # Only header exists
             print("No data collected to save.")
             return
        try:
            columns = data[0]
            data_rows = data[1:]
            df = pd.DataFrame(data_rows, columns=columns)
            csv_filename = "output_play.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved successfully to {csv_filename}")
        except Exception as e:
            print(f"Error saving data to {csv_filename}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play Overcooked manually or with a simple AI partner.")
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help='Grid world size.')
    parser.add_argument('--task_name', type=str, default='tomato salad', choices=TASKLIST, help='The recipe/task name.')
    parser.add_argument('--map_type', type=str, default="A", choices=MAPTYPES_TRAIN, help='The type of map layout.')
    parser.add_argument('--mode', type=str, default="vector", choices=["vector", "image"], help='Environment observation mode (vector recommended).')
    parser.add_argument('--debug', action="store_true", help='Enable debug prints in environment.')
    parser.add_argument('--agent_partner_type', type=str, default='human', choices=['human', 'stationary', 'random'], help='Type of AI agent partner.') # Add 'trained' later

    args = parser.parse_args()

    # Create and run the Player instance
    try:
        player = Player(
            grid_dim=args.grid_dim,
            task_name=args.task_name,
            map_type=args.map_type,
            mode=args.mode,
            debug=args.debug,
            agent_partner_type=args.agent_partner_type
        )
        player.run()
    except ValueError as e:
        print(f"Error during setup: {e}")
    except ImportError as e:
         print(f"Error importing required modules: {e}")
    except Exception as e_main:
         print(f"An unexpected error occurred: {e_main}")
         traceback.print_exc()

