# Multi-Agent Overcooked Environment with Adaptive Agents

This project implements a multi-agent reinforcement learning environment based on the game Overcooked. The primary goal is to train an AI agent ('ai') capable of coordinating with a second agent ('human') under varying incentive structures, fostering the emergence of complex adaptive strategies through self-play.

N.B. The current training is very spammy due to debug, consider adding a flag to suppress for prints.
## Environment Overview

* **Based on Overcooked:** Simulates the core mechanics of Overcooked on a customizable grid (`--grid_dim`). Includes ingredients (tomato, lettuce, onion), tools (knives, plates), counters, and delivery spots.
* **Multi-Agent:** Features two agents, identified as 'ai' and 'human'.
* **Tasks:** Agents collaborate (or compete) to complete specific salad recipes (e.g., "tomato salad", "lettuce-onion-tomato salad") defined by the `--task` argument.
* **Observations:** Uses vector-based observations representing the state of all items, agents, the task, and partner information. Frame stacking (`--frame_stack_size`) is supported for temporal reasoning.
* **Actions:** Discrete action space including movement (Up, Down, Left, Right) and Interact.
* **Reward Shaping:** Includes sparse rewards for task completion (delivery) and penalties for errors, plus denser shaping rewards for achieving useful sub-goals (picking up needed raw items, chopping needed items, plating needed items) to guide learning.

## Key Feature: Partner Role Simulation & Adaptive Learning

A core feature of this project is the ability to simulate different partner dynamics and train agents to adapt:

1.  **Configurable Partner Role:** The `human_role` parameter can be set during training or testing ([`helpful`, `adversarial`, `neutral`, `random_h_a`, `random_h_n_a`]).
2.  **Reward Multiplier:** This role determines a `human_multiplier` applied to the base step reward for the 'human' agent each episode:
    * `helpful`: Multiplier = `1.0` (Aligned incentives)
    * `adversarial`: Multiplier = `-1.0` (Opposed incentives)
    * `neutral`: Multiplier = `0.0` (Indifferent partner)
    * `random_h_a`: Multiplier is randomly `1.0` or `-1.0` per episode.
    * `random_h_n_a`: Multiplier is randomly `1.0`, `0.0`, or `-1.0` per episode.
3.  **Observed Context Signal:** The AI agent **explicitly observes** information about the current `human_multiplier` (both a one-hot encoding of the role type and the numerical multiplier value itself) as part of its input state vector.
4.  **Self-Play Training:** The primary training method is now **self-play**, where *both* the 'ai' and 'human' policies are trained simultaneously using PPO. During this process, the `human_multiplier` is typically randomized (`random_h_n_a`) per episode. This forces both learning agents to constantly adapt to partners with potentially different (and changing) incentives.
5.  **Enhanced Observations:** Agents also observe their partner's normalized position and whether the partner is currently holding an item, providing more context for coordination or counter-play.

The combination of self-play, randomized incentives, and observation of the partner's state/role promotes the emergence of robust and adaptive strategies rather than policies overfit to a single partner type.

## Recent Updates

*(Based on user-provided draft and conversation)*

### Map & Environment
* Map layouts (5x5, potentially others like 9x9 if defined) should ideally include all vegetable types for complex tasks.
* Added handling for `--grid_dim` argument in training script.
* Corrected environment `reset` logic for consistent role/multiplier assignment.
* Added reward shaping for pickup, chopping, and plating needed items.
* Included partner position and holding status in observations.

## Training and Testing

### Training (Self-Play for Emergence)

To train both agents simultaneously to co-evolve adaptive strategies using the `random_h_n_a` role setting and enhanced observations (example uses 9x9 map from user draft, adjust task/map/grid as needed):

``bash
python train_rllib.py \
  --name overcooked_selfplay_9x9_A_v1 \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --human_role "random_h_n_a" \
  --frame_stack_size 4 \
  --lr 5e-5 \
  --stop_iters 2000 \
  --checkpoint_freq 100 \
  --save_dir "my_overcooked_runs"
``

**Key parameters for self-play training:**
* `--name`: Descriptive name for the run.
* `--task`: Choose a sufficiently complex task (e.g., multi-ingredient).
* `--grid_dim`, `--map_type`: Define the environment layout.
* `--human_role "random_h_n_a"`: **Crucial** for forcing agents to learn adaptive policies based on varying partner incentives.
* `--frame_stack_size`: Number of frames to stack (e.g., 4).
* `--lr`: Learning rate (e.g., `5e-5`, adjust based on stability).
* `--stop_iters`: Number of training iterations (likely needs to be high for self-play, e.g., 2000+).
* `--checkpoint_freq`: How often to save checkpoints.
* **Note:** Ensure `train_rllib.py` is configured for self-play (`policies_to_train=['ai', 'human']`) and the desired PPO parameters (like `entropy_coeff`, likely hardcoded low, e.g., 0.01) are set in `define_training`.

### Testing Trained Policies

After training, use `run_trained.py` to load the checkpoint and evaluate the learned policies. You can test how the agents behave when faced with a *specific* partner type during evaluation by setting the `--human_role` for the test run.

```bash
# Test with partner acting HELPFUL (multiplier = 1.0)
# Load the checkpoint from the self-play run name prefix
python run_trained.py \
  --save_dir "my_overcooked_runs" \
  --name "overcooked_selfplay_9x9_A_v1" \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --frame_stack_size 4 \
  --human_role "helpful" \
  --num_episodes 5 \
  --max_steps 300 \
  --sleep 0.2
```

```bash
# Test with partner acting ADVERSARIAL (multiplier = -1.0)
# Load the checkpoint from the self-play run name prefix
python run_trained.py \
  --save_dir "my_overcooked_runs" \
  --name "overcooked_selfplay_9x9_A_v1" \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --frame_stack_size 4 \
  --human_role "adversarial" \
  --num_episodes 5 \
  --max_steps 300 \
  --sleep 0.2
```

**Key parameters for testing:**
* `--save_dir`, `--name`: Identify the training run to load the checkpoint from. Alternatively, use `--experiment_path` to specify the exact run directory.
* `--task`, `--grid_dim`, `--map_type`, `--frame_stack_size`: Should generally match the configuration the agent was trained on.
* `--human_role`: Set to `helpful`, `adversarial`, or `neutral` to evaluate the agent's conditional behavior when faced with that specific partner incentive structure during the test.
* `--num_episodes`: Number of evaluation episodes.
* `--max_steps`: Maximum steps per evaluation episode (should match environment setting).
* `--sleep`: Delay for visualization.

*(Observe the rendered gameplay and console output to analyze how the 'ai' agent's strategy changes (or doesn't) based on the fixed `human_role` set during testing, reflecting its ability to adapt based on the observed context signal).* 

### Other Scripts
* `play.py`: Allows a human to play the game, controlling one agent alongside a simple baseline partner (or potentially another human if modified).
* *(Mention `debug_render.py` here if it exists and explain its purpose).*

## TODOs:
* *(Easy):*
* Consider randomization in initial setup if this makes sense (probably increases the number of needed training iterations)
* Add complexity to the map/environment to get rid of trivial behaviors like dropping and picking up items or just holding onto an item
  * Add more usable resources to map?
  * Add some level of game dynamics that make this level of behavior not suitable, e.g., penalize cyclic behavior

* *(Difficult):*
* Parametrize the concept of "trust" somehow 
  * Could this be tied to the observation space? Heuristics?
  * Number of consecutive steps? Increase frame stacking length, replace with a NN?
* Track/log trust evolving (easy)
* Add Bayesian methods for parametrized trust (probably makes the adapting much quicker, but computationally more expensive)