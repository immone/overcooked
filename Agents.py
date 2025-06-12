from collections import defaultdict

import numpy as np
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

class AlwaysStationaryRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [4] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]
        
    @override(RLModule)
    def get_state(self, **kwargs):
        # Return a simple state dict for the stationary agent
        return {"state": "stationary"}


class RandomRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [self.action_space.sample()] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]
        
    @override(RLModule)
    def get_state(self, **kwargs):
        # Return a simple state dict for the random agent
        return {"state": "random"}
    
class ObstructingRLM(RLModule):
    """Human agent that intentionally obstructs the AI agent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstruction_chance = 0.6  # 60% chance to obstruct
        
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        actions = []
        for obs in batch[Columns.OBS]:
            if np.random.random() < self.obstruction_chance:
                # Get AI position from observation and move toward it
                ai_pos = self._extract_ai_position(obs)
                action = self._move_toward_position(ai_pos)
            else:
                # Otherwise take random action
                action = np.random.randint(0, 6)  # Assuming 6 actions
            actions.append(action)
        return {Columns.ACTIONS: np.array(actions)}
    
    def _extract_ai_position(self, obs):
        # Extract AI position from observation
        # This is a simplification - you'll need to adjust based on your observation space
        return np.array([2, 2])  # Default position
        
    def _move_toward_position(self, target_pos):
        # Move toward target position
        # Simplified logic - replace with actual movement logic
        directions = [1, 2, 3, 4]  # Up, right, down, left
        return np.random.choice(directions)
    
    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError("ObstructingRLM is not trainable")
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

    def forward_train(self, batch):
        raise NotImplementedError("ObstructingRLM only supports inference")

    def forward_inference(self, batch):
        # Generate actions that tend to move toward AI agent (simplified)
        batch_size = batch["obs"].shape[0]
        # Mostly move (actions 1-4) to get in the way
        actions = torch.randint(1, 5, (batch_size,))
        return {"action": actions}
        
    # Add these methods to fix checkpointing error
    def get_state(self, *args, **kwargs):
        # Simple RLModules don't have meaningful state to save
        return {}
        
    def set_state(self, state):
        # Nothing to restore for this simple module
        pass


class ClumsyRLM(RLModule):
    """Human agent that frequently makes mistakes with objects"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mistake_chance = 0.6  # 60% chance to make mistakes
        
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        actions = []
        for obs in batch[Columns.OBS]:
            if np.random.random() < self.mistake_chance:
                # Make a mistake - pick up or drop objects at wrong places
                # Picking up (action 5) or interacting with objects randomly
                action = 5  # Pick up/drop action
            else:
                # Otherwise move randomly
                action = np.random.randint(0, 5)  # Movement actions
            actions.append(action)
        return {Columns.ACTIONS: np.array(actions)}
    
    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError("ClumsyRLM is not trainable")
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

    def forward_train(self, batch):
        raise NotImplementedError("ClumsyRLM only supports inference")

    def forward_inference(self, batch):
        # Generate actions that tend to interact with objects (pickup/drop)
        batch_size = batch["obs"].shape[0]
        # 60% chance to interact with objects (action 5)
        mask = torch.rand(batch_size) < 0.6
        actions = torch.randint(0, 5, (batch_size,))  # Movement actions
        actions[mask] = 5  # Set to interact action where mask is True
        return {"action": actions}
    
    # Add these methods to fix checkpointing error
    def get_state(self, *args, **kwargs):
        # Simple RLModules don't have meaningful state to save
        return {}
        
    def set_state(self, state):
        # Nothing to restore for this simple module
        pass


class SelfishRLM(RLModule):
    """Human agent that tries to complete the task without helping AI"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_goal = None
        
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        actions = []
        for obs in batch[Columns.OBS]:
            # Try to identify ingredients and complete task independently
            # In a real implementation, this would run a simple planning algorithm
            # For now, simulate goal-directed behavior with random actions
            if self.current_goal is None or np.random.random() < 0.1:
                self.current_goal = np.random.randint(0, 3)  # 0=get ingredient, 1=cut, 2=deliver
                
            if self.current_goal == 0:
                action = np.random.choice([1, 2, 3, 4, 5])  # Move + pickup
            elif self.current_goal == 1:
                action = np.random.choice([1, 2, 3, 4, 5])  # Move + interact
            else:
                action = np.random.choice([1, 2, 3, 4, 5])  # Move + deliver
                
            actions.append(action)
        return {Columns.ACTIONS: np.array(actions)}
    
    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError("SelfishRLM is not trainable")
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

    def forward_train(self, batch):
        raise NotImplementedError("SelfishRLM only supports inference")

    def forward_inference(self, batch):
        # Try to do task alone with goal-directed behavior
        batch_size = batch["obs"].shape[0]
        # Mix of movement and interaction
        actions = torch.randint(0, 6, (batch_size,))
        return {"action": actions}
    
    # Add these methods to fix checkpointing error
    def get_state(self, *args, **kwargs):
        # Simple RLModules don't have meaningful state to save
        return {}
        
    def set_state(self, state):
        # Nothing to restore for this simple module
        pass

class HelpfulRLM(RLModule):
    """Human agent that coordinates with AI to complete tasks faster"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtask_focus = np.random.randint(0, 2)  # 0=get ingredients, 1=cut and deliver
        
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        actions = []
        for obs in batch[Columns.OBS]:
            # Every 5 steps, switch subtask focus to complement AI
            if np.random.random() < 0.2:
                self.subtask_focus = 1 - self.subtask_focus
                
            if self.subtask_focus == 0:
                # Focus on getting ingredients - move around and pick up
                action = np.random.choice([1, 2, 3, 4, 5])  # Move + pickup
            else:
                # Focus on cutting and delivering
                action = np.random.choice([1, 2, 3, 4, 5])  # Move + interact
                
            actions.append(action)
        return {Columns.ACTIONS: np.array(actions)}
    
    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError("HelpfulRLM is not trainable")
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

    def forward_train(self, batch):
        raise NotImplementedError("HelpfulRLM only supports inference")

    def forward_inference(self, batch):
        # Generate actions that complement AI (simplified)
        batch_size = batch["obs"].shape[0]
        # Mix of all actions with emphasis on productive ones
        actions = torch.randint(1, 6, (batch_size,))  # Avoid staying still
        return {"action": actions}
    
    # Add these methods to fix checkpointing error
    def get_state(self, *args, **kwargs):
        # Simple RLModules don't have meaningful state to save
        return {}
        
    def set_state(self, state):
        # Nothing to restore for this simple module
        pass