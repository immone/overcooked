from collections import defaultdict
import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from unittest.mock import Mock
from environment.items import Tomato, Lettuce, Onion, Plate, Knife, Agent, Food


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
            "AlwaysStationaryRLM is not trainable. NEVER include it in `config.multi_agent(policies_to_train={...})` set."
            
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


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
            "AlwaysStationaryRLM is not trainable. NEVER include it in `config.multi_agent(policies_to_train={...})` set."
           
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


class PartnerRLM(RLModule):
    """
    Partner agent that mixes cooperative heuristic, sabotage heuristic, and random actions
    based on a quality score q ∈ [-1, 1].
    
    TODO: Phase 3 - Environment Integration
    - quality_score will be set via environment's info dict: info['partner_q']
    - This will happen in train_adaptive.py through a custom callback
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = 0.05  # Noise floor
        self.quality_score = 0.0  # Default neutral
        self.cooperative_heuristic = None
        self.sabotage_heuristic = None
        self._initialized = False
        self.obs_parser = ObservationParser(grid_dim=(5, 5))  # TODO: Phase 3 - get from config
        
    def set_quality_score(self, q):
        """Set the partner's quality score q ∈ [-1, 1]"""
        self.quality_score = np.clip(q, -1.0, 1.0)
        
    def _initialize_heuristics(self):
        """Lazy initialization of heuristics"""
        if not self._initialized:
            from heuristics import CooperativeHeuristic, SabotageHeuristic
            self.cooperative_heuristic = CooperativeHeuristic()
            self.sabotage_heuristic = SabotageHeuristic()
            self._initialized = True
    
    def _compute_mixture_probabilities(self, q):
        """
        Compute mixture probabilities based on quality score q.
        
        Returns:
            tuple: (p_help, p_sab, p_noise)
        """
        q_clipped = np.clip(q, -1.0, 1.0)
        
        # Mixture probabilities from the proposal
        p_noise = self.epsilon + (1 - self.epsilon) * (1 - abs(q_clipped))
        p_help = (1 - self.epsilon) * max(0, q_clipped)
        p_sab = (1 - self.epsilon) * max(0, -q_clipped)
        
        # Ensure probabilities sum to 1 (should be automatic, but let's be safe)
        total = p_help + p_sab + p_noise
        if total > 0:
            p_help /= total
            p_sab /= total
            p_noise /= total
            
        return p_help, p_sab, p_noise
    
    def _get_partner_agent_index(self):
        """Get the index of the partner agent (human)"""
        # Partner controls the "human" agent which is index 0
        return 0
    
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        """Forward pass for inference - samples from mixture policy"""
        self._initialize_heuristics()
        
        obs_batch = batch[Columns.OBS]
        batch_size = len(obs_batch)
        actions = []
        
        # Get mixture probabilities
        p_help, p_sab, p_noise = self._compute_mixture_probabilities(self.quality_score)
        
        for i in range(batch_size):
            obs = obs_batch[i]
            
            # Parse observation to reconstruct environment state
            mock_env = self.obs_parser.parse_observation(obs, agent_id="human")
            
            # Sample which policy to use
            rand = np.random.random()
            
            if rand < p_help and self.cooperative_heuristic is not None:
                # Use cooperative heuristic
                action = self.cooperative_heuristic.get_action(mock_env, self._get_partner_agent_index())
            elif rand < p_help + p_sab and self.sabotage_heuristic is not None:
                # Use sabotage heuristic
                action = self.sabotage_heuristic.get_action(mock_env, self._get_partner_agent_index())
            else:
                # Use random action
                action = self.action_space.sample()
                
            actions.append(action)
            
        return {Columns.ACTIONS: np.array(actions)}
    
    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        # Same as inference for this non-learning agent
        return self._forward_inference(batch, **kwargs)
    
    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "PartnerRLM is not trainable! It uses fixed heuristics based on quality score."
        )
    
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]
    
    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]
    
    def reset(self):
        """Reset any internal state of heuristics"""
        if self.cooperative_heuristic is not None:
            self.cooperative_heuristic.reset()
        if self.sabotage_heuristic is not None:
            self.sabotage_heuristic.reset()


class ObservationParser:
    """
    Parses flat observation vector into a mock environment structure
    that heuristics can work with.
    """
    
    def __init__(self, grid_dim=(5, 5)):
        self.xlen, self.ylen = grid_dim
        # Add default map layout for counters
        # TODO: Phase 3 - Get actual map layout from environment config
        self._init_default_map_layout()

    
        
    def parse_observation(self, obs_vector, agent_id="human"):
        """
        Parse observation vector into mock environment object.
        
        The observation vector format (from Overcooked._initObs):
        [tomato.x, tomato.y, tomato.status, lettuce.x, lettuce.y, lettuce.status, 
         onion.x, onion.y, onion.status, plate1.x, plate1.y, plate2.x, plate2.y,
         knife1.x, knife1.y, knife2.x, knife2.y, delivery.x, delivery.y,
         agent1.x, agent1.y, agent2.x, agent2.y, onehotTask...]
        """
        # Create mock environment
        mock_env = Mock()
        mock_env.xlen = self.xlen
        mock_env.ylen = self.ylen
        
        # Initialize containers
        mock_env.itemDic = {
            "tomato": [],
            "lettuce": [],
            "onion": [],
            "plate": [],
            "knife": [],
            "delivery": [],
            "agent": []
        }
        
        # Parse observation - positions are normalized, need to denormalize
        idx = 0
        
        # Parse food items (3 items with status)
        food_types = [
            ("tomato", Tomato),
            ("lettuce", Lettuce),
            ("onion", Onion)
        ]
        
        for food_name, food_class in food_types:
            if idx + 2 < len(obs_vector):
                x = int(obs_vector[idx] * self.xlen)
                y = int(obs_vector[idx + 1] * self.ylen)
                status = obs_vector[idx + 2]
                
                # Create mock food item
                item = Mock(spec=food_class)
                item.x = x
                item.y = y
                item.rawName = food_name
                item.chopped = (status >= 1.0)  # Fully chopped when status = 1
                item.cur_chopped_times = int(status * 3)  # Assuming 3 chops required
                item.required_chopped_times = 3
                
                mock_env.itemDic[food_name].append(item)
                idx += 3
        
        # Parse plates (2 plates, no status)
        for i in range(2):
            if idx + 1 < len(obs_vector):
                x = int(obs_vector[idx] * self.xlen)
                y = int(obs_vector[idx + 1] * self.ylen)
                
                plate = Mock(spec=Plate)
                plate.x = x
                plate.y = y
                plate.containing = None  # TODO: Phase 3 - parse plate contents from extended obs
                plate.rawName = "plate"
                
                mock_env.itemDic["plate"].append(plate)
                idx += 2
        
        # Parse knives (2 knives, no status)
        for i in range(2):
            if idx + 1 < len(obs_vector):
                x = int(obs_vector[idx] * self.xlen)
                y = int(obs_vector[idx + 1] * self.ylen)
                
                knife = Mock(spec=Knife)
                knife.x = x
                knife.y = y
                knife.holding = None  # TODO: Phase 3 - parse knife contents from extended obs
                knife.rawName = "knife"
                
                mock_env.itemDic["knife"].append(knife)
                idx += 2
        
        # Parse delivery (1 delivery point)
        if idx + 1 < len(obs_vector):
            x = int(obs_vector[idx] * self.xlen)
            y = int(obs_vector[idx + 1] * self.ylen)
            
            delivery = Mock()
            delivery.x = x
            delivery.y = y
            delivery.rawName = "delivery"
            
            mock_env.itemDic["delivery"].append(delivery)
            idx += 2
        
        # Parse agents (2 agents)
        agents = []
        for i in range(2):
            if idx + 1 < len(obs_vector):
                x = int(obs_vector[idx] * self.xlen)
                y = int(obs_vector[idx + 1] * self.ylen)
                
                agent = Mock(spec=Agent)
                agent.x = x
                agent.y = y
                agent.holding = None  # TODO: Phase 3 - parse held items from extended obs
                agent.color = ["blue", "robot"][i]  # Assuming human is blue, ai is robot
                
                agents.append(agent)
                idx += 2
        
        mock_env.agent = agents
        mock_env.itemDic["agent"] = agents
        
        # Parse task (remaining values are one-hot task encoding)
        task_encoding = obs_vector[idx:] if idx < len(obs_vector) else []
        mock_env.task = self._decode_task(task_encoding)
        
        # Create mock map (simplified - just mark item positions)
        mock_env.map = [[0 for _ in range(self.ylen)] for _ in range(self.xlen)]
        
        # Mark positions on map
        for item_list in mock_env.itemDic.values():
            for item in item_list:
                if 0 <= item.x < self.xlen and 0 <= item.y < self.ylen:
                    if hasattr(item, 'rawName'):
                        if item.rawName == "knife":
                            mock_env.map[item.x][item.y] = 6
                        elif item.rawName == "delivery":
                            mock_env.map[item.x][item.y] = 7
                        # TODO: Phase 3 - properly reconstruct full map with counters
                    
        return mock_env
    
    def _decode_task(self, task_encoding):
        """Decode one-hot task encoding back to task string"""
        # TODO: Phase 3 - This will be properly set from environment
        # For now, return a default task
        tasks = [
            "tomato salad", "lettuce salad", "onion salad",
            "lettuce-tomato salad", "onion-tomato salad", 
            "lettuce-onion salad", "lettuce-onion-tomato salad"
        ]
        
        if len(task_encoding) >= len(tasks):
            for i, val in enumerate(task_encoding[:len(tasks)]):
                if val > 0.5:  # One-hot encoded
                    return tasks[i]
        
        return "tomato salad"  # Default
    
    def _init_default_map_layout(self):
        """Initialize a default map layout with counters"""
        # For 5x5 map, add some counters (this is a placeholder)
        # Real map will come from environment in Phase 3
        self.default_counters = {
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # Top row
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),  # Bottom row
        }
