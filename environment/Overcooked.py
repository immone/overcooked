import numpy as np
from environment.render.game import Game
from gymnasium import spaces
from .items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import Counter
import random 

DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX = {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
AGENTCOLOR = ["blue", "robot", "green", "yellow"]
TASKLIST = [
    "tomato salad", "lettuce salad", "onion salad",
    "lettuce-tomato salad", "onion-tomato salad",
    "lettuce-onion salad", "lettuce-onion-tomato salad"
]
MAPTYPES_TRAIN = ["A", "B", "C"]
class Overcooked_multi(MultiAgentEnv):
    """
    A Multi-Agent Environment for the Overcooked game based on the provided code,
    incorporating fixes for reward calculation, initialization, Gymnasium API compatibility,
    and detailed debug printing.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_dim, task, rewardList, map_type="A", mode="vector", debug=False, human_role="helpful", frame_stack_size=4):
        super().__init__()
        self.step_count = 0
        self.agents = self.possible_agents = ["human", "ai"]
        self.n_agent = len(self.agents)
        self.obs_radius = 0
        self.xlen, self.ylen = grid_dim
        self.task = task
        self.mapType = map_type
        self.debug = debug 
        self.mode = mode
        self.human_role = human_role
        self.frame_stack_size = frame_stack_size
        

        self.rewardList = rewardList

        self.rewardList.setdefault("pickup_needed_raw", 1.0)
        self.rewardList.setdefault("chopped_needed", 2.0)
        self.rewardList.setdefault("plated_needed_chopped", 3.0)
        self.rewardList.setdefault("correct_delivery", 200.0)
        self.rewardList.setdefault("wrong_delivery", -50.0)
        self.rewardList.setdefault("metatask_failed", -5.0)
        self.rewardList.setdefault("subtask_finished", 0.0)
        self.rewardList.setdefault("step_penalty", -0.01)
        self.rewardList.setdefault("exploration", 0.01)
        self.rewardList.setdefault("movement_reward", 0.1)


        if self.mode != "vector":
             print("Warning: This environment heavily assumes vector observations. Image mode might not work correctly.")

        self.reward = None
        self.reward_stats = {}
        self.human_multiplier = 1.0
        self.episode_role_type = self.human_role

        self.initMap = self._initialize_map()
        if not self.initMap:
             raise ValueError(f"Could not initialize map for grid_dim={grid_dim}, mapType={map_type}, n_agent={self.n_agent}")
        self.map = copy.deepcopy(self.initMap)
        self.oneHotTask = [1 if t == self.task else 0 for t in TASKLIST]

        self.taskCompletionStatus = []

        self._createItems()

        self.action_spaces = {agent_id: spaces.Discrete(5) for agent_id in self.agents}

        try:
            dummy_obs_dict = self._get_obs(add_multiplier=True)
            single_agent_obs = dummy_obs_dict[self.agents[0]]
            single_obs_dim = single_agent_obs.shape[0]
            stacked_obs_dim = single_obs_dim * self.frame_stack_size
            # Use float32 for compatibility with RLlib/PyTorch/TF
            obs_space = spaces.Box(low=-1.0, high=1.0, shape=(stacked_obs_dim,), dtype=np.float32)
            self.observation_spaces = {agent_id: obs_space for agent_id in self.agents}
        except Exception as e:
            print(f"Error determining observation space: {e}. Check _get_obs and _get_vector_state.")
            raise e

        # Observation history for frame stacking
        self.obs_history = {agent_id: [] for agent_id in self.agents}

        self.game = None
        try:
             self.game = Game(self)
        except Exception as e:
             print(f"Warning: Failed to initialize Game rendering during __init__: {e}")



    def _initialize_map(self):
        if self.xlen == 3 and self.ylen == 3:
            return self._map_3x3()
        elif self.xlen == 5 and self.ylen == 5:
            return self._map_5x5()
        elif self.xlen == 3 and self.ylen == 5:
            return self._map_3x5()
        elif self.xlen == 7 and self.ylen == 7:
            return self._map_7x7()
        elif self.xlen == 9 and self.ylen == 9:
            return self._map_9x9()
        else:
            return []
    
    def _map_3x3(self):
        map = None
        if self.n_agent == 2:
            if self.mapType == "A":
                map = [[8, 3, 4],
                    [7, 2, 6],
                    [1, 5, 2]]
            elif self.mapType == "B":
                map = [[1, 3, 8],
                    [7, 2, 6],
                    [1, 5, 4]]
            elif self.mapType == "C":
                map = [[4, 3, 8],
                    [7, 2, 6],
                    [1, 5, 2]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map = [[8, 3, 2],
                    [7, 2, 6],
                    [4, 5, 2]]
            elif self.mapType == "B":
                map = [[8, 3, 2],
                    [7, 2, 6],
                    [4, 5, 2]]
            elif self.mapType == "C":
                map = [[8, 3, 2],
                    [7, 2, 6],
                    [4, 5, 2]]
        return map

    def _map_5x5(self):
        map = None
        if self.n_agent == 2:
            if self.mapType == "A":
                map = [[1, 1, 1, 1, 1],
                    [6, 0, 0, 2, 1],
                    [3, 0, 0, 0, 1],
                    [7, 0, 4, 2, 1],
                    [1, 5, 8, 1, 1]]
            elif self.mapType == "B":
                map = [[1, 8, 1, 1, 1],
                    [6, 2, 1, 0, 1],
                    [3, 0, 5, 2, 6],
                    [7, 0, 5, 0, 1],
                    [1, 4, 1, 1, 1]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 5, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 0, 6],
                    [7, 0, 4, 8, 1],
                    [1, 1, 1, 1, 1]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map = [[1, 1, 5, 1, 1],
                    [6, 2, 0, 2, 1],
                    [3, 0, 0, 0, 6],
                    [7, 0, 2, 0, 1],
                    [8, 4, 5, 1, 1]]
            elif self.mapType == "B":
                map = [[1, 1, 1, 1, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 2, 6],
                    [7, 0, 5, 0, 1],
                    [8, 4, 1, 1, 1]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 5, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 0, 6],
                    [7, 2, 0, 0, 1],
                    [8, 4, 1, 1, 1]]
        return map

    def _map_3x5(self):
        map = None
        if self.n_agent == 2:
            if self.mapType == "A":
                map = [[1, 1, 1, 1, 1],
                    [6, 2, 0, 2, 8],
                    [3, 0, 0, 0, 1],
                    [7, 0, 0, 0, 1],
                    [1, 5, 4, 1, 1]]
            elif self.mapType == "B":
                map = [[1, 8, 1, 1, 1],
                    [6, 2, 1, 0, 1],
                    [3, 0, 5, 2, 4]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 5, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 0, 6],
                    [7, 0, 4, 8, 1],
                    [1, 1, 1, 1, 1]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map = [[1, 1, 5, 1, 1],
                    [6, 2, 0, 2, 1],
                    [3, 0, 0, 0, 6],
                    [7, 0, 2, 0, 1],
                    [8, 4, 5, 1, 1]]
            elif self.mapType == "B":
                map = [[1, 1, 1, 1, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 2, 6],
                    [7, 0, 5, 0, 1],
                    [8, 4, 1, 1, 1]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 5, 1],
                    [6, 2, 1, 2, 1],
                    [3, 0, 5, 0, 6],
                    [7, 2, 0, 0, 1],
                    [8, 4, 1, 1, 1]]
        return map
        
    def _map_7x7(self):
        map = None
        if self.n_agent == 2:
            if self.mapType == "A":
                map = [[1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 0, 0, 0, 4],
                    [6, 0, 0, 0, 0, 0, 8],
                    [7, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 2, 0, 0, 0, 5],
                    [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map = [[1, 5, 1, 0, 1, 4, 1],
                    [1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 4, 1, 1, 0, 1],
                    [1, 0, 2, 1, 2, 0, 1],
                    [7, 0, 0, 1, 0, 0, 5],
                    [1, 6, 8, 3, 7, 6, 1]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 1, 2, 0, 4],
                    [6, 0, 0, 1, 0, 0, 8],
                    [7, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 5],
                    [1, 1, 1, 1, 1, 5, 1]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map = [[1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 0, 2, 0, 4],
                    [6, 0, 0, 0, 0, 0, 8],
                    [7, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 2, 0, 0, 0, 5],
                    [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map = [[1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 1, 2, 0, 4],
                    [6, 0, 0, 1, 0, 0, 8],
                    [7, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 2, 1, 0, 0, 5],
                    [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map = [[1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 1, 2, 0, 4],
                    [6, 0, 0, 1, 0, 0, 8],
                    [7, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 2, 0, 0, 0, 5],
                    [1, 1, 1, 1, 1, 5, 1]]
        return map

    def _map_9x9(self):
        if self.mapType == "A":
            map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],  # 1=Counter, 3=Tomato Source
                    [6, 0, 2, 0, 0, 0, 2, 0, 4],  # 6=Knife, 0=Space, 2=Agent, 4=Lettuce Source
                    [6, 0, 0, 0, 0, 0, 0, 0, 8],  # 8=Onion Source
                    [7, 0, 0, 0, 0, 0, 0, 0, 1],  # 7=Delivery
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 5],  # 5=Plate Source
                    [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        elif self.mapType == "B":
            map =  [[1, 1, 5, 1, 0, 1, 1, 4, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 2, 0, 1, 0, 1, 0, 2, 1], # Added agents here for example
                    [1, 0, 0, 4, 1, 7, 0, 0, 1], # Added lettuce/delivery
                    [1, 0, 6, 0, 1, 0, 6, 0, 1], # Added knives
                    [7, 0, 0, 0, 1, 0, 0, 0, 1], # Added delivery
                    [1, 3, 1, 1, 1, 8, 5, 1, 1]] # Added tomato/onion/plate
        elif self.mapType == "C":
            map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                    [6, 0, 2, 0, 1, 0, 2, 0, 4],
                    [6, 0, 0, 0, 1, 0, 0, 0, 8],
                    [7, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 5],
                    [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        else:
             print(f"Warning: Invalid mapType '{self.mapType}' for 9x9 grid. Falling back to map A.")
             map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                     [6, 0, 2, 0, 0, 0, 2, 0, 4],
                     [6, 0, 0, 0, 0, 0, 0, 0, 8],
                     [7, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 5],
                     [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        return map

    def _get_stacked_obs(self):
        current_obs_dict = self._get_obs()
        stacked_obs_dict = {}

        for agent_id in self.agents:
            current_agent_obs = current_obs_dict[agent_id]

            if not self.obs_history[agent_id]:
                self.obs_history[agent_id] = [current_agent_obs.copy()] * self.frame_stack_size
            else:
                self.obs_history[agent_id].append(current_agent_obs.copy())
                if len(self.obs_history[agent_id]) > self.frame_stack_size:
                    self.obs_history[agent_id].pop(0)

            history = list(self.obs_history[agent_id])
            while len(history) < self.frame_stack_size:
                history.insert(0, history[0].copy())

            stacked_obs_dict[agent_id] = np.concatenate(history).astype(np.float32)

        return stacked_obs_dict

    def _createItems(self):
        self.itemDic = {name: [] for name in ITEMNAME if name not in ["space", "counter"]}
        agent_idx = 0
        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.onion = []
        self.plate = []

        if not self.map: # Check if map is valid
             print("Error: Cannot create items, map is empty.")
             return

        for x in range(self.xlen):
            for y in range(self.ylen):
                # Bounds check for map access
                if x < len(self.map) and y < len(self.map[x]):
                    item_type_idx = self.map[x][y]
                    if 0 <= item_type_idx < len(ITEMNAME):
                        item_type = ITEMNAME[item_type_idx]
                        # Create items based on type
                        if item_type == "agent":
                            if agent_idx < self.n_agent and agent_idx < len(AGENTCOLOR):
                                agent_instance = Agent(x, y, color=AGENTCOLOR[agent_idx])
                                self.itemDic[item_type].append(agent_instance)
                                agent_idx += 1
                            else: print(f"Warning: Skipped agent creation at ({x},{y}). Index {agent_idx} exceeds n_agent {self.n_agent} or available colors {len(AGENTCOLOR)}.")
                        elif item_type == "knife":
                            new_knife = Knife(x, y); self.itemDic[item_type].append(new_knife); self.knife.append(new_knife)
                        elif item_type == "delivery":
                            new_delivery = Delivery(x, y); self.itemDic[item_type].append(new_delivery); self.delivery.append(new_delivery)
                        elif item_type == "tomato":
                            new_tomato = Tomato(x, y); self.itemDic[item_type].append(new_tomato); self.tomato.append(new_tomato)
                        elif item_type == "lettuce":
                            new_lettuce = Lettuce(x, y); self.itemDic[item_type].append(new_lettuce); self.lettuce.append(new_lettuce)
                        elif item_type == "onion":
                            new_onion = Onion(x, y); self.itemDic[item_type].append(new_onion); self.onion.append(new_onion)
                        elif item_type == "plate":
                            new_plate = Plate(x, y); self.itemDic[item_type].append(new_plate); self.plate.append(new_plate)
                    else: print(f"Warning: Invalid item index {item_type_idx} in map at ({x},{y})")
                else: print(f"Warning: Map access out of bounds during item creation at ({x},{y})")

        self.itemList = [item for sublist in self.itemDic.values() for item in sublist]
        self.agent = self.itemDic.get("agent", [])

        if len(self.agent) != self.n_agent:
            print(f"FATAL ERROR: Number of agent instances created ({len(self.agent)}) does not match expected n_agent ({self.n_agent}). Check map definition.")

    def _initObs(self):
        """
        Initialize the observations for the agents.

        This function creates an observation list by normalizing the positions of items
        and appending additional information if the item is of type Food. It then extends
        the observation list with one-hot encoded task information. Finally, it assigns
        the observation list to each agent and returns a list of observations for all agents.

        Returns:
            list: A list containing the observation arrays for each agent.
        """
        obs = []
        for item in self.itemList:
            obs.extend([item.x / self.xlen, item.y / self.ylen])
            if isinstance(item, Food):
                obs.append(item.cur_chopped_times / item.required_chopped_times)
        obs.extend(self.oneHotTask)

        for agent in self.agent:
            agent.obs = obs
        return [np.array(obs)] * self.n_agent

    def _get_vector_state(self, add_multiplier=True):
        state_list = []
        agent_states = {}

        # 1. Item States
        for item in self.itemList:
            x = item.x / self.xlen
            y = item.y / self.ylen
            state_list.extend([x, y])
            if isinstance(item, Food): state_list.append(item.cur_chopped_times / item.required_chopped_times if hasattr(item,'required_chopped_times') and item.required_chopped_times > 0 else 0.0)
            elif isinstance(item, Plate): state_list.append(1.0 if hasattr(item, 'containing') and item.containing else 0.0)
            elif isinstance(item, Knife): state_list.append(1.0 if hasattr(item, 'holding') and item.holding else 0.0)
            elif isinstance(item, Agent):
                agent_states[self.agents[self.agent.index(item)]] = { 
                    'x': x, 'y': y,
                    'holding_bool': 1.0 if hasattr(item, 'holding') and item.holding else 0.0,
                    'holding_item': getattr(item.holding, 'rawName', None) if hasattr(item, 'holding') and item.holding else None
                }
                state_list.append(1.0 if hasattr(item, 'holding') and item.holding else 0.0)

        # 2. Task Encoding
        state_list.extend(self.oneHotTask)

        # 3. Role/Multiplier Encoding
        role_encoding = [0.0, 0.0, 0.0]; current_multiplier = 0.0
        if hasattr(self, 'human_multiplier'):
            current_multiplier = self.human_multiplier
            if current_multiplier > 0: role_encoding[0] = 1.0
            elif current_multiplier < 0: role_encoding[1] = 1.0
            else: role_encoding[2] = 1.0
        else: role_encoding[2] = 1.0
        state_list.extend(role_encoding)
        state_list.append(float(current_multiplier))

        partner_states = []
        agent_ids = list(self._agent_ids) # e.g., ["human", "ai"]
        for i, agent_id in enumerate(agent_ids):
            partner_id = agent_ids[1-i] # Get the ID of the other agent
            if partner_id in agent_states:
                partner_state = agent_states[partner_id]
                partner_states.extend([
                    partner_state['x'],
                    partner_state['y'],
                    partner_state['holding_bool']
                ])
                item_held = partner_state['holding_item'],
                holding_one_hot = [1.0 if name == item_held else 0.0 for name in ITEMNAME if name not in ['agent','space','counter']]
                partner_states.extend(holding_one_hot)
            else:
                print(f"Warning: Could not find state for partner {partner_id}")
                partner_states.extend([0.0, 0.0, 0.0])

        state_list.extend(partner_states)
        final_state = np.array(state_list, dtype=np.float32)
        # print(f"Debug: Final state vector length = {len(final_state)}") # Debug length
        return final_state

    def _get_image_state(self):
        """
        Retrieve the current image state for each agent.

        This method returns a list containing the current image observation 
        of the game state, repeated for each agent in the environment.

        Returns:
            list: A list where each element is the current image observation 
              of the game state, repeated for the number of agents.
        """
        return [self.game.get_image_obs()] * self.n_agent

    def _get_obs(self, add_multiplier=True):
        if self.mode != "vector":
            raise NotImplementedError("Image mode not fully supported.")
        if self.obs_radius != 0:
            raise NotImplementedError("Partial observability vector mode not fully supported.")

        # Get the global state vector
        base_obs_vector = self._get_vector_state(add_multiplier=add_multiplier)

        # Create the observation dictionary for multi-agent API
        obs_dict = {agent_id: base_obs_vector.copy() for agent_id in self.agents}
        return obs_dict

    def _get_vector_obs(self):

        """
        Returns
        -------
        vector_obs : list
            vector observation for each agent.
        """

        po_obs = []
        for agent in self.agent:
            obs = []
            idx = 0


            if self.xlen == 3 and self.ylen == 3:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
            elif self.xlen == 5 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 3 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 7 and self.ylen == 7:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap= [[1, 1, 1, 0, 1, 1, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
            elif self.xlen == 9 and self.ylen == 9:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    
                    agent.pomap= [[1, 1, 1, 1, 0, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 1, 1, 0, 0, 1],
                                  [1, 0, 2, 0, 1, 0, 2, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

            for item in self.itemList:
                # Check if the item is within the agent's observation radius or if the radius is 0 (full observability)
                if (agent.x - self.obs_radius <= item.x <= agent.x + self.obs_radius and
                    agent.y - self.obs_radius <= item.y <= agent.y + self.obs_radius) or self.obs_radius == 0:
                    # Normalize item position and add to observation
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status to observation
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)
                        idx += 1
                else:
                    # If the item is outside the observation radius, use its initial position
                    x = agent.obs[idx] * self.xlen
                    y = agent.obs[idx + 1] * self.ylen
                    if not (agent.x - self.obs_radius <= x <= agent.x + self.obs_radius and
                            agent.y - self.obs_radius <= y <= agent.y + self.obs_radius):
                        x = item.initial_x
                        y = item.initial_y
                    x /= self.xlen
                    y /= self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status from the agent's previous observation
                    if isinstance(item, Food):
                        obs.append(agent.obs[idx] / item.required_chopped_times)
                        idx += 1
                # Update the agent's partial observability map
                agent.pomap[int(x * self.xlen)][int(y * self.ylen)] = ITEMIDX[item.rawName]
            # Mark the agent's position on the partial observability map
            agent.pomap[agent.x][agent.y] = ITEMIDX["agent"]
            # Add one-hot encoded task information to the observation
            obs.extend(self.oneHotTask)
            agent.obs = obs
            po_obs.append(np.array(obs))
        return po_obs

    def _get_image_obs(self):

        """
        Returns
        -------
        image_obs : list
            image observation for each agent.
        """

        po_obs = []
        frame = self.game.get_image_obs()
        old_image_width, old_image_height, channels = frame.shape
        new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
        new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
        color = (0,0,0)
        obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame

        for idx, agent in enumerate(self.agent):
            agent_obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)
            po_obs.append(agent_obs)
        return po_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def _findItem(self, x, y, itemName):
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    @property
    def state_size(self):
        return self.get_state().shape[0]

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]
    
    def _update_coordination_metrics(self, actions, positions):
        """Track and reward coordination behavior"""
        
        if positions["human"] and positions["ai"]:
            human_pos = positions["human"]
            ai_pos = positions["ai"]

        # More sophisticated tracking could be added here
        return self.coordination_events

    def _initialize_coordination_tracking(self):
        """Track coordination between agents for trust learning"""
        self.coordination_events = {
            "ai_following_human_lead": 0,
            "complementary_actions": 0,
            "conflicting_actions": 0,
            "successful_handoffs": 0,
        }
        
        # Previous agent positions/actions
        self.previous_positions = {agent: None for agent in self.agents}
        self.previous_actions = {agent: None for agent in self.agents}

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    def _add_reward(self, reward_type, amount=None):
        """Helper to add reward and track it by type. Handles None reward."""
        # Ensure self.reward is initialized (should be done in step start)
        if self.reward is None:
             print(f"FATAL ERROR: _add_reward called when self.reward is None! Step:{self.step_count}, Type:{reward_type}")
             self.reward = 0.0

        if amount is None:
            amount = float(self.rewardList.get(reward_type, 0.0))
        else:
             amount = float(amount)

        # Log before adding for clarity
        reward_before_adding = self.reward
        print(f"DEBUG Reward Step {self.step_count}: Adding Type='{reward_type}', Amount={amount:.4f}. Step Reward BEFORE add: {reward_before_adding:.4f}", end="")

        self.reward += amount
        print(f", AFTER add: {self.reward:.4f}")

        if not hasattr(self, 'reward_stats'): self.reward_stats = {}
        if reward_type not in self.reward_stats: self.reward_stats[reward_type] = 0.0
        self.reward_stats[reward_type] += amount

        return amount
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.map = copy.deepcopy(self.initMap)
        self._createItems()
        self.step_count = 0

        self.reward_stats = {
            "step_penalty": 0.0,
            "subtask_finished": 0.0,
            "correct_delivery": 0.0,
            "wrong_delivery": 0.0,
            "metatask_failed": 0.0,
            "pickup_needed_raw": 0.0, # New specific reward
            "chopped_needed": 0.0,    # New specific reward
            "plated_needed_chopped": 0.0,
        }

        counter = Counter(self.task)
        self.taskCompletionStatus = [counter[element] if element in counter else 0 for element in TASKLIST]

        self.episode_role_type = self.human_role

        if self.human_role == "helpful":
            self.human_multiplier = 1.0
        elif self.human_role == "adversarial":
            self.human_multiplier = -1.0
        elif self.human_role == "neutral":
            self.human_multiplier = 0.0
        elif self.human_role == "random_h_a":
            self.human_multiplier = float(random.choice([1, -1]))
            self.episode_role_type = "random_helpful" if self.human_multiplier > 0 else "random_adversarial"
        elif self.human_role == "random_h_n_a":
            self.human_multiplier = float(random.choice([1, 0, -1]))
            if self.human_multiplier > 0: self.episode_role_type = "random_helpful"
            elif self.human_multiplier < 0: self.episode_role_type = "random_adversarial"
            else: self.episode_role_type = "random_neutral"
        else:
            print(f"Warning: Unknown human_role '{self.human_role}'. Defaulting to helpful.")
            self.human_multiplier = 1.0
            self.episode_role_type = "helpful"

        standard_obs = self._get_obs()

        self.obs_history = {agent: [standard_obs[agent].copy()] for agent in self.agents}

        if self.frame_stack_size > 1:
             initial_stacked_obs = self._get_stacked_obs()
             return initial_stacked_obs, {}
        else:
             return standard_obs, {}
    
    def _debug_rewards(self, base_reward):
        """Return tracked reward components for debugging"""
        debug_info = self.reward_stats.copy()
        debug_info["total_step_reward_calculated"] = float(base_reward)
        return debug_info

    
    def _track_reward_components(self):
        """Create a data structure to track reward components for debugging"""
        return {
            "total": 0.0,
            "step_penalty": 0.0,
            "subtask_finished": 0.0,
            "metatask_failed": 0.0,
            "correct_delivery": 0.0,
            "wrong_delivery": 0.0,
            "coordination_bonus": 0.0,
            "implicit_bonuses": 0.0
        }
    
    def step(self, action_dict: dict):
        """ Executes one environment step for all agents. """

        step_penalty = float(self.rewardList.get("step penalty", -0.01))
        self.reward = step_penalty

        if not hasattr(self, 'reward_stats'): self.reward_stats = {}
        self.reward_stats["step_penalty"] = self.reward_stats.get("step_penalty", 0.0) + step_penalty


        self.step_count += 1
        print(f"\n--- Step {self.step_count} (Role: {self.episode_role_type}) ---")

        if not isinstance(action_dict, dict):
             print(f"Warning: Step received non-dict action {action_dict}. Defaulting.")
             action_dict = {agent_id: 4 for agent_id in self.agents} # Default to 'stay'
        # Get action for each agent from the dict, defaulting to 'stay'
        action_list = [action_dict.get(agent_id, 4) for agent_id in self.agents]

        if any(a != 4 for a in action_list):
             self._add_reward("exploration", self.rewardList.get("exploration", 0.01))

        pickup_reward = float(self.rewardList.get("pickup_needed_raw", 0.0))
        chop_reward = float(self.rewardList.get("chopped_needed", 0.0))
        plate_reward = float(self.rewardList.get("plated_needed_chopped", 0.0))
        correct_delivery_reward = float(self.rewardList.get("correct_delivery", 0.0))
        wrong_delivery_penalty = float(self.rewardList.get("wrong_delivery", 0.0))
        metatask_fail_penalty = float(self.rewardList.get("metatask_failed", 0.0))

        step_specific_info = {'collision': []} # Info specific to this step (e.g., collisions)
        done = False # Flag for task completion this step

        # Sequential execution (process one agent's action at a time)
        for idx, agent_id in enumerate(self.agents):
            if idx >= len(self.agent):
                 print(f"Error: Agent index {idx} out of bounds for self.agent list (len {len(self.agent)})")
                 continue
            agent = self.agent[idx] # Get the Agent object instance
            agent_action = action_list[idx] # Get the action for this agent
            print(f"DEBUG Step {self.step_count}: Agent {idx} ({agent_id}) attempting Action {agent_action}")

            if 0 <= agent_action < 4:
                target_x = agent.x + DIRECTION[agent_action][0]
                target_y = agent.y + DIRECTION[agent_action][1]

                if not (0 <= target_x < self.xlen and 0 <= target_y < self.ylen):
                    print(f"DEBUG Step {self.step_count} Agent {idx}: Move out of bounds.")
                    continue

                target_obj_idx = self.map[target_x][target_y]
                target_name = ITEMNAME[target_obj_idx]

                if target_name == "agent":
                    print(f"DEBUG Step {self.step_count} Agent {idx}: Collision with agent.")
                    step_specific_info['collision'].append(agent_id)
                    continue # Cannot move into another agent
                elif target_name != "space":
                    print(f"DEBUG Step {self.step_count} Agent {idx}: Move blocked by {target_name}.")
                    continue # Cannot move into walls, counters, items etc.
                else:
                    print(f"DEBUG Step {self.step_count} Agent {idx}: Moving to ({target_x},{target_y})")
                    self.map[agent.x][agent.y] = ITEMIDX["space"] # Old position becomes space
                    agent.move(target_x, target_y) # Update agent's internal position
                    self.map[agent.x][agent.y] = ITEMIDX["agent"] # New position shows agent

            elif agent_action == 4:
                interaction_occurred_this_agent = False
                for dx, dy in DIRECTION:
                    target_x = agent.x + dx
                    target_y = agent.y + dy

                    # Check bounds for target square
                    if not (0 <= target_x < self.xlen and 0 <= target_y < self.ylen):
                        continue # Skip if adjacent square is out of bounds

                    # Get name of object at target square
                    target_obj_idx = self.map[target_x][target_y]
                    target_name = ITEMNAME[target_obj_idx]

                    # Cannot interact with empty space or other agents via action 4
                    if target_name in ["space", "agent"]:
                        continue

                    if target_name == "counter":
                        print(f"DEBUG Step {self.step_count} Agent {idx}: Trying INTERACT with COUNTER at ({target_x},{target_y}). Agent holding: {agent.holding}")
                        if agent.holding:
                             # Try to place held item onto the counter
                             held_item_name = getattr(agent.holding, 'rawName', 'unknown_item')
                             # Check if the counter tile on the map is actually clear (index 1)
                             if self.map[target_x][target_y] == ITEMIDX["counter"]:
                                 if held_item_name in ITEMIDX: # Check if item type is valid on map
                                      print(f"DEBUG Step {self.step_count} Agent {idx}: Putting down {held_item_name} on counter.")
                                      self.map[target_x][target_y] = ITEMIDX[held_item_name] # Update map state
                                      agent.putdown(target_x, target_y) # Agent releases item
                                      interaction_occurred_this_agent = True; break # Interaction done
                                 else:
                                      print(f"Warning: Cannot put down {held_item_name}, not in ITEMIDX.")
                                      self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                             else: # Counter tile is occupied
                                  occupying_item = ITEMNAME[self.map[target_x][target_y]]
                                  print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot putdown, counter occupied by {occupying_item}.")
                                  self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                        else:
                             # Interacting with counter while empty-handed does nothing
                             print(f"DEBUG Step {self.step_count} Agent {idx}: Interacted with empty counter (no effect).")
                             interaction_occurred_this_agent = True; break # Counts as interaction attempt, stop checking directions

                    # Handle interaction with other items (Plate, Knife, Food, Delivery)
                    else:
                        # Find the actual item instance at the target location
                        item = self._findItem(target_x, target_y, target_name)

                        # Safety check if item instance wasn't found (shouldn't happen if map is consistent)
                        if item is None and target_name != "delivery": # Delivery might not be an 'item' instance
                            print(f"Warning: Map indicates {target_name} at ({target_x},{target_y}) but _findItem returned None.")
                            continue # Skip interaction attempt for this direction

                        print(f"DEBUG Step {self.step_count} Agent {idx}: Trying INTERACT with {target_name} at ({target_x},{target_y}). Agent holding: {agent.holding}")

                        # Case 1: Agent is NOT holding anything
                        if not agent.holding:
                            if target_name in ["tomato", "lettuce", "onion", "plate"] and item:
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Trying PICKUP {target_name}")
                                agent.pickup(item) # Agent now holds the item
                                self.map[target_x][target_y] = ITEMIDX["counter"] # Location becomes counter tile
                                item_name = getattr(item, 'rawName', 'Item') # Get name for logging
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Picked up {item_name}")
                                # Add shaping reward if applicable
                                if isinstance(item, Food) and not item.chopped:
                                    task_ingredients = self.task.replace(' salad','').split('-')
                                    if item.rawName in task_ingredients:
                                        self._add_reward("pickup_needed_raw", pickup_reward)
                                interaction_occurred_this_agent = True; break # Action done

                            elif target_name == "knife" and item:
                                knife = item
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Trying INTERACT with knife (empty handed)")
                                if isinstance(knife.holding, Food): # Knife has food
                                    food_on_knife = knife.holding
                                    if food_on_knife.chopped: # Pick up chopped food
                                        print(f"DEBUG Step {self.step_count} Agent {idx}: Picking up chopped {food_on_knife.rawName} from knife")
                                        item_to_pickup = knife.release() # Assumes release returns item
                                        agent.pickup(item_to_pickup)
                                        interaction_occurred_this_agent = True; break
                                    else: # Chop the food on the knife
                                        print(f"DEBUG Step {self.step_count} Agent {idx}: Chopping {food_on_knife.rawName} on knife")
                                        food_on_knife.chop() # Assumes chop method exists
                                        if food_on_knife.chopped:
                                             print(f"DEBUG Step {self.step_count} Agent {idx}: Finished chopping {food_on_knife.rawName}")
                                             task_ingredients = self.task.replace(' salad','').split('-')
                                             if food_on_knife.rawName in task_ingredients:
                                                 self._add_reward("chopped_needed", chop_reward)
                                        interaction_occurred_this_agent = True; break
                                elif isinstance(knife.holding, Plate): # Pick up plate from knife
                                    print(f"DEBUG Step {self.step_count} Agent {idx}: Picking up plate from knife")
                                    item_to_pickup = knife.release()
                                    agent.pickup(item_to_pickup)
                                    interaction_occurred_this_agent = True; break
                                else: # Knife is empty
                                    print(f"DEBUG Step {self.step_count} Agent {idx}: Interacted with empty knife (no effect).")
                                    interaction_occurred_this_agent = True; break # Counts as interaction attempt
                            # Add other interactions for empty-handed agent if needed (e.g., delivery?)

                        # Case 2: Agent IS holding something 
                        elif agent.holding:
                            held_item = agent.holding # Reference to the item agent is holding
                            # Interaction logic when agent is holding item
                            if target_name == "plate" and item:
                                plate = item
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Trying PLATE {held_item.rawName}")
                                if isinstance(held_item, Food) and held_item.chopped:
                                    # Check if item already on plate (prevents duplicate adds/rewards)
                                    already_on_plate = plate.containing and hasattr(held_item, 'rawName') and held_item.rawName in [getattr(f, 'rawName', None) for f in plate.containing]
                                    if not already_on_plate:
                                        task_ingredients = self.task.replace(' salad','').split('-')
                                        is_needed = hasattr(held_item, 'rawName') and held_item.rawName in task_ingredients
                                        print(f"DEBUG Step {self.step_count} Agent {idx}: Plating {held_item.rawName} onto plate.")
                                        # Agent puts down item, plate contains it
                                        agent.putdown(target_x, target_y) # Item logically moves to plate
                                        plate.contain(held_item) # Assumes Plate.contain method works
                                        if is_needed: self._add_reward("plated_needed_chopped", plate_reward)
                                        interaction_occurred_this_agent = True; break
                                    else: print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot plate, {getattr(held_item,'rawName','Item')} already on plate.")
                                elif isinstance(held_item, Food) and not held_item.chopped:
                                    print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot plate raw food.")
                                    self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                                else: # Trying to plate non-food or invalid item
                                    print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot plate item {held_item}")
                                    self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break

                            elif target_name == "knife" and item:
                                knife = item
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Trying INTERACT with knife while holding {getattr(held_item,'rawName','Item')}")
                                if not knife.holding: # Knife empty -> Try placing unchopped food on it
                                    if isinstance(held_item, Food) and not held_item.chopped:
                                         print(f"DEBUG Step {self.step_count} Agent {idx}: Placing {held_item.rawName} on knife")
                                         agent.putdown(target_x, target_y) # Item moves to knife location
                                         knife.hold(held_item) # Assumes Knife.hold method exists
                                         interaction_occurred_this_agent = True; break
                                    else: # Cannot place chopped food or non-food on empty knife
                                         print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot place {getattr(held_item,'rawName','Item')} on empty knife.")
                                         self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                                elif isinstance(knife.holding, Food): # Knife has food
                                     if isinstance(held_item, Plate): # Holding plate -> Try plating from knife
                                          food_on_knife = knife.holding
                                          if food_on_knife.chopped:
                                               # Check if already on plate
                                               already_on_plate = held_item.containing and hasattr(food_on_knife,'rawName') and food_on_knife.rawName in [getattr(f, 'rawName', None) for f in held_item.containing]
                                               if not already_on_plate:
                                                   task_ingredients = self.task.replace(' salad','').split('-')
                                                   is_needed = hasattr(food_on_knife,'rawName') and food_on_knife.rawName in task_ingredients
                                                   print(f"DEBUG Step {self.step_count} Agent {idx}: Plating {food_on_knife.rawName} from knife")
                                                   plate_content = knife.release() # Knife releases item
                                                   held_item.contain(plate_content) # Plate (held_item) contains it
                                                   if is_needed: self._add_reward("plated_needed_chopped", plate_reward)
                                                   interaction_occurred_this_agent = True; break
                                               else: print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot plate from knife, {getattr(food_on_knife,'rawName','Item')} already on plate.")
                                          else: # Knife has raw food
                                               print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot plate raw food from knife.")
                                               self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                                     else: # Agent holding something else (e.g., food) -> Invalid action
                                          print(f"DEBUG Step {self.step_count} Agent {idx}: Invalid interaction: Holding {getattr(held_item,'rawName','Item')}, knife has {getattr(knife.holding,'rawName','Item')}")
                                          self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break
                                # Add cases for knife holding plate if needed
                                else: # Knife has plate or unknown state
                                     print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot interact with knife (knife state: {knife.holding})")
                                     self._add_reward("metatask_failed", metatask_fail_penalty); interaction_occurred_this_agent = True; break

                            elif target_name == "delivery": # No item instance needed for delivery spot usually
                                print(f"DEBUG Step {self.step_count} Agent {idx}: Attempting DELIVER item {held_item}")
                                if isinstance(held_item, Plate) and hasattr(held_item, 'containing') and held_item.containing:
                                    plate = held_item
                                    dishName = getattr(plate, 'containedName', '')
                                    task_name_for_check = self.task
                                    print(f"DEBUG Step {self.step_count} Agent {idx}: Delivering plate with '{dishName}', Task is '{task_name_for_check}'")

                                    is_correct_dish = False
                                    try:
                                        # Example: Compare sets of ingredient names (case-insensitive)
                                        # Requires containedName format like "tomato-lettuce"
                                        dish_parts = set(d.strip().lower() for d in dishName.replace('Chopped','').replace('Fresh','').split('-') if d)
                                        task_parts = set(t.strip().lower() for t in task_name_for_check.replace(' salad','').split('-') if t)
                                        print(f"DEBUG Dish Check: Dish Parts={dish_parts}, Task Parts={task_parts}") # Debug the check itself
                                        if dish_parts and task_parts and dish_parts == task_parts:
                                            is_correct_dish = True
                                    except AttributeError:
                                         print(f"Warning: Could not parse dishName '{dishName}' during delivery check.")

                                    if is_correct_dish:
                                        print(f"DEBUG Step {self.step_count} Agent {idx}: Dish IS correct.")
                                        try:
                                            task_index = TASKLIST.index(task_name_for_check)
                                            if self.taskCompletionStatus[task_index] > 0:
                                                self.taskCompletionStatus[task_index] -= 1
                                                self._add_reward("correct_delivery", correct_delivery_reward)
                                                print(f"DEBUG Step {self.step_count} Agent {idx}: CORRECT DELIVERY! Task status: {self.taskCompletionStatus}")
                                                done = all(value == 0 for value in self.taskCompletionStatus) # Check if all tasks done

                                                # Refresh delivered items
                                                foods_delivered = plate.containing; agent.putdown(target_x, target_y)
                                                plate.release(); plate.refresh(); self.map[plate.x][plate.y] = ITEMIDX["plate"]
                                                for f in foods_delivered: f.refresh(); self.map[f.x][f.y] = ITEMIDX.get(f.rawName, ITEMIDX["counter"]) # Use get for safety
                                                interaction_occurred_this_agent = True; break
                                            else: # Delivered correct dish, but task already completed
                                                print(f"DEBUG Step {self.step_count} Agent {idx}: Correct dish delivered, but task count was already 0.")
                                                self._add_reward("wrong_delivery", wrong_delivery_penalty)
                                                # Refresh items
                                                foods_delivered = plate.containing; agent.putdown(target_x, target_y)
                                                plate.release(); plate.refresh(); self.map[plate.x][plate.y] = ITEMIDX["plate"]
                                                for f in foods_delivered: f.refresh(); self.map[f.x][f.y] = ITEMIDX.get(f.rawName, ITEMIDX["counter"])
                                                interaction_occurred_this_agent = True; break
                                        except ValueError: # Task name not found in TASKLIST
                                            print(f"Warning: Task '{task_name_for_check}' not in TASKLIST during delivery reward check.")
                                            self._add_reward("wrong_delivery", wrong_delivery_penalty)
                                            interaction_occurred_this_agent = True; break
                                    else: # Dish incorrect
                                        print(f"DEBUG Step {self.step_count} Agent {idx}: Dish IS WRONG.")
                                        self._add_reward("wrong_delivery", wrong_delivery_penalty)
                                        # Refresh items
                                        plate_to_reset = agent.holding; foods_on_plate = plate_to_reset.containing
                                        agent.putdown(target_x, target_y); plate_to_reset.release(); plate_to_reset.refresh()
                                        self.map[plate_to_reset.x][plate_to_reset.y] = ITEMIDX["plate"]
                                        if foods_on_plate:
                                            for f in foods_on_plate: f.refresh(); self.map[f.x][f.y] = ITEMIDX.get(f.rawName, ITEMIDX["counter"])
                                        interaction_occurred_this_agent = True; break

                                elif isinstance(held_item, Plate) and not (hasattr(held_item, 'containing') and held_item.containing):
                                     print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot deliver empty plate.")
                                     self._add_reward("wrong_delivery", wrong_delivery_penalty)
                                     plate_to_reset = agent.holding; agent.putdown(target_x, target_y)
                                     plate_to_reset.refresh(); self.map[plate_to_reset.x][plate_to_reset.y] = ITEMIDX["plate"]
                                     interaction_occurred_this_agent = True; break
                                else: # Cannot deliver this item type
                                     print(f"DEBUG Step {self.step_count} Agent {idx}: Cannot deliver item {held_item}.")
                                     self._add_reward("wrong_delivery", wrong_delivery_penalty)
                                     item_to_reset = agent.holding; agent.putdown(target_x, target_y)
                                     item_to_reset.refresh() # Assumes refresh method exists
                                     # Try to put item icon back on map (might fail if pos occupied)
                                     try:
                                         if self.map[item_to_reset.x][item_to_reset.y] == ITEMIDX["counter"]:
                                             self.map[item_to_reset.x][item_to_reset.y] = ITEMIDX.get(getattr(item_to_reset,'rawName',None), ITEMIDX["counter"])
                                     except: pass # Ignore error if refresh fails map update
                                     interaction_occurred_this_agent = True; break

                    # If an interaction happened, break direction loop for this agent
                    if interaction_occurred_this_agent:
                        break

                # If action was 4 but loop finished without break -> No interaction happened
                if not interaction_occurred_this_agent:
                     print(f"DEBUG Step {self.step_count} Agent {idx}: Interact action (4) had no valid target/effect.")

        max_steps = 300 # Define max steps per episode
        timed_out = self.step_count >= max_steps
        # terminated: Task completed successfully
        terminated = done
        # truncated: Episode ended due to time limit, but not by task completion
        truncated = timed_out and not terminated

        terminateds = {agent_id: terminated for agent_id in self._agent_ids}
        truncateds = {agent_id: truncated for agent_id in self._agent_ids}
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        if terminated: print(f"--- EPISODE TERMINATED (Task Complete) at Step {self.step_count} ---")
        if truncated: print(f"--- EPISODE TRUNCATED (Timeout) at Step {self.step_count} ---")

        # self.reward holds the base reward accumulated this step
        final_rewards = {
            agent_id: float(self.reward) * (self.human_multiplier if agent_id == "human" else 1.0)
            for agent_id in self.agents
        }
        # Ensure floats
        final_rewards = {k: float(v) for k, v in final_rewards.items()}
        print(f"DEBUG Step {self.step_count} End: Base Step Reward={self.reward:.4f}, Multiplier={self.human_multiplier}, Final Step Rewards={final_rewards}")

        next_obs = self._get_stacked_obs() if self.frame_stack_size > 1 else self._get_obs()

        infos = {}
        episode_end = terminateds["__all__"] or truncateds["__all__"]
        for agent_id in self.agents:
            agent_info = {}
            # Add step-specific info (e.g., collisions)
            agent_info.update(step_specific_info) # Add collisions etc.

            if episode_end:
                print(f"--- Logging Episode End Info for Agent {agent_id} ---")
                # Add all tracked reward stats
                for key, value in self.reward_stats.items():
                     agent_info[f'episode_cumulative_{key}'] = value
                # Add other summary info
                agent_info['episode_human_multiplier'] = self.human_multiplier
                agent_info['episode_role_type_str'] = str(self.episode_role_type)
                agent_info['episode_length'] = self.step_count
                agent_info['episode_task_complete'] = done # Task completion status

            infos[agent_id] = agent_info

        return next_obs, final_rewards, terminateds, truncateds, infos


    def render(self, mode='human'):
        return self.game.on_render()

    





