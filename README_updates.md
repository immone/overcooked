## Recent Updates

### Map Improvements
- All map layouts have been updated to include all three vegetable types (tomato, lettuce, onion) for each map configuration (A, B, C).
- Added 9x9 map layouts (`_map_9x9` method) for larger environment experiments.

### Code Fixes
- Fixed observation stacking mechanism for temporal reasoning.
- Corrected map initialization to ensure consistent map layouts, including adding missing `_map_9x9` definition.
- Corrected `reset` method in `Overcooked.py` to correctly set the `human_multiplier` based on the `human_role` argument (helpful=1, adversarial=-1, random=random).
- Added error handling for image loading to prevent crashes with missing assets.
- Improved plate handling with null checking to prevent errors.
- Fixed multiple minor bugs throughout the codebase.
- Added `--grid_dim` argument handling to `train_rllib.py`.

### New Features
- **Human Context Signal & Role Setting**:
    - The environment can be configured via `human_role` (`helpful`, `adversarial`, `random`).
    - This setting determines the `human_multiplier` (1.0 for helpful, -1.0 for adversarial, random choice for random) set at the start of each episode.
    - The AI agent **observes this `human_multiplier` value** as part of its input vector.
    - This allows the AI to learn adaptive strategies conditioned on this explicit context signal.
- **Frame Stacking**: Support for temporal observations through frame stacking.

## Training and Testing the Adaptive AI

### Training the AI to Adapt
To train the AI agent to adapt its behavior based on the observed `human_multiplier` signal (using the large 9x9 map):


```bash
python train_rllib.py \
  --name adaptive_ai_9x9_A_v1 \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --human_role "random" \
  --frame_stack_size 4 \
  --stop_iters 500 \
  --checkpoint_freq 50
```

Key parameters for training adaptive AI:
* `--name`: A descriptive name for your training run (e.g., `adaptive_ai_9x9_A_v1`).
* `--task`: Recipe to train on.
* `--grid_dim`: Grid dimensions (e.g., `9 9`).
* `--map_type`: Map layout to use (e.g., `A`).
* `--human_role "random"`: **Essential for training.** Ensures the AI experiences both multiplier values (1 and -1) during training episodes.
* `--frame_stack_size`: Number of frames to stack.
* `--stop_iters`: Number of training iterations (consider increasing for larger maps).
* `--checkpoint_freq`: How often to save checkpoints.

*(Note: The training script currently trains only the 'ai' policy).*

### Testing the Trained Adaptive AI
To test how the trained AI behaves conditioned on the context signal it observes:


```bash
# Test with helpful context (multiplier = 1.0 observed by AI)
python run_trained.py \
  --name adaptive_ai_9x9_A_v1 \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --frame_stack_size 4 \
  --human_role "helpful" \
  --num_episodes 5
```


```bash
# Test with adversarial context (multiplier = -1.0 observed by AI)
python run_trained.py \
  --name adaptive_ai_9x9_A_v1 \
  --task "lettuce-onion-tomato salad" \
  --grid_dim 9 9 \
  --map_type "A" \
  --frame_stack_size 4 \
  --human_role "adversarial" \
  --num_episodes 5
```

Key parameters for testing:
* `--name`: The base name used during training (e.g., `adaptive_ai_9x9_A_v1`). Or use `--experiment_path` for the full path.
* `--task`, `--grid_dim`, `--map_type`, `--frame_stack_size`: Must match the training configuration of the loaded model.
* `--human_role`: Set to `helpful` or `adversarial` to fix the multiplier the AI observes during the test run and evaluate its conditional behavior. Set to `random` to see behavior with random context switching per episode.
* `--num_episodes`: Number of test episodes.

*(Observe the AI's actions to see if its strategy changes based on the context signal (`human_multiplier`) it receives).*

### Debugging Environment Issues
For diagnosing environment issues or testing specific configurations (like checking map layouts), run `debug_render.py`.