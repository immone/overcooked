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

    def __init__(self, grid_dim, task, rewardList, map_type="A", mode="vector", debug=False, human_role="helpful", frame_stack_size=4):
        super().__init__()
        self.step_count = 0
        self.agents = self.possible_agents = ["human", "ai"]
        self.n_agent = len(self.agents)
        self.obs_radius = 0
        self.xlen, self.ylen = grid_dim

        self.task = task
        self.rewardList = rewardList
        self.mapType = map_type
        self.debug = debug
        self.mode = mode
        if self.mode != "vector":
             print("Warning: This modification assumes vector observations. Image mode might not work correctly.")

        # Store human role setting (though we use multiplier directly in obs)
        self.human_role = human_role
        self.frame_stack_size = frame_stack_size

        self.reward = None
        self.human_multiplier = 1 # Initialize, will be randomized in reset

        self.initMap = self._initialize_map()
        self.map = copy.deepcopy(self.initMap)

        self.oneHotTask = [1 if t in self.task else 0 for t in TASKLIST]

        counter = Counter(self.task)
        self.taskCompletionStatus = [counter[element] if element in counter else 0 for element in TASKLIST]

        self._createItems() # Creates self.agent, self.itemList etc.
        # Note: self.n_agent is already set earlier

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}

        # --- Observation Space Modification ---
        # Calculate the base observation dimension *before* frame stacking
        # We get a dummy observation first to determine its length
        dummy_base_obs_dict = self._get_obs(add_multiplier=True) # Pass flag to include multiplier dim
        single_obs_dim = len(dummy_base_obs_dict[self.agents[0]])

        # Now define the observation space including the potential frame stacking
        stacked_obs_dim = single_obs_dim * frame_stack_size
        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(stacked_obs_dim,), dtype=np.float64)
            # Changed low bound to -1 for the multiplier
            for agent in self.agents
        }
        # --- End Observation Space Modification ---

        # Initialize history after defining obs space
        self.obs_history = {agent: [] for agent in self.agents}

        # Initialize game rendering
        self.game = Game(self)


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
        current_obs = self._get_obs()
        
        for agent in self.agents:
            # Ensure history is initialized if empty (e.g., first call after reset)
            if not self.obs_history[agent]:
                 # Pad with current obs if history is shorter than stack size
                 self.obs_history[agent] = [current_obs[agent].copy()] * self.frame_stack_size
            else:
                 self.obs_history[agent].append(current_obs[agent].copy())
                 if len(self.obs_history[agent]) > self.frame_stack_size:
                     self.obs_history[agent].pop(0)

        # Create stacked observations
        stacked_obs = {}
        for agent in self.agents:
            # Pad if needed (e.g. first few steps)
            history = list(self.obs_history[agent])
            while len(history) < self.frame_stack_size:
                history.insert(0, history[0]) # Duplicate oldest frame

            # Concatenate the observations
            stacked_obs[agent] = np.concatenate(history)

        return stacked_obs

    def _createItems(self):
        """
        Initialize the items in the environment based on the map configuration.

        This function iterates through the map grid and creates instances of various items
        (e.g., agents, knives, delivery points, tomatoes, lettuce, onions, plates) at their
        respective positions. It populates the item dictionary and item list with these instances.

        The item dictionary (itemDic) maps item names to lists of item instances, excluding "space" and "counter".
        The item list (itemList) is a flattened list of all item instances.
        The agent list (agent) is a list of all agent instances.

        The function also assigns colors to agents based on their index.

        Returns:
            None
        """
        self.itemDic = {name: [] for name in ITEMNAME if name != "space" and name != "counter"}
        agent_idx = 0

        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.onion = []
        self.plate = []

        for x in range(self.xlen):
            for y in range(self.ylen):
                item_type = ITEMNAME[self.map[x][y]]
                if item_type == "agent":
                    self.itemDic[item_type].append(Agent(x, y, color=AGENTCOLOR[agent_idx]))
                    agent_idx += 1
                elif item_type == "knife":
                    new_knife = Knife(x, y)
                    self.itemDic[item_type].append(new_knife)
                    self.knife.append(new_knife)
                elif item_type == "delivery":
                    new_delivery = Delivery(x, y)
                    self.itemDic[item_type].append(new_delivery)
                    self.delivery.append(new_delivery)
                elif item_type == "tomato":
                    new_tomato = Tomato(x, y)
                    self.itemDic[item_type].append(new_tomato)
                    self.tomato.append(new_tomato)
                elif item_type == "lettuce":
                    new_lettuce = Lettuce(x, y)
                    self.itemDic[item_type].append(new_lettuce)
                    self.lettuce.append(new_lettuce)
                elif item_type == "onion":
                    new_onion = Onion(x, y)
                    self.itemDic[item_type].append(new_onion)
                    self.onion.append(new_onion)
                elif item_type == "plate":
                    new_plate = Plate(x, y)
                    self.itemDic[item_type].append(new_plate)
                    self.plate.append(new_plate)

        self.itemList = [item for sublist in self.itemDic.values() for item in sublist]
        self.agent = self.itemDic["agent"]

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

    def _get_vector_state(self, add_multiplier=False):
        """
        Get the global state of the environment.

        This function creates a global state representation by normalizing the positions of items
        and appending additional information based on the item type. It includes the positions of
        all items, their chopped status if they are food, whether plates contain food, and whether
        knives or agents are holding items. The state is extended with one-hot encoded task information.

        Returns:
            list: A list containing the state arrays for each agent.
        """
        state = []
        for item in self.itemList:
            x = item.x / self.xlen
            y = item.y / self.ylen
            state.extend([x, y])

            if isinstance(item, Food):
                state.append(item.cur_chopped_times / item.required_chopped_times)
            elif isinstance(item, Plate):
                state.append(1.0 if item.containing else 0.0)
            elif isinstance(item, Knife):
                state.append(1.0 if item.holding else 0.0)
            elif isinstance(item, Agent):
                state.append(1.0 if item.holding else 0.0)

        state.extend(self.oneHotTask)
        if add_multiplier:
             # Add the multiplier value (-1 or 1) as the last feature
             state.append(float(self.human_multiplier))
             
        return np.array(state, dtype=np.float64)

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
        """
        Returns observation for each agent. Adds human_multiplier if add_multiplier=True.
        Assumes vector mode for this modification.
        """
        if self.mode != "vector":
            return self._get_image_state() if self.obs_radius == 0 else self._get_image_obs()

        if self.obs_radius > 0:
             # TODO: If using partial observability, modify _get_vector_obs
             print("Warning: Partial observability vector mode not explicitly updated for multiplier.")
             base_obs_vector = self._get_vector_state(add_multiplier=add_multiplier)

        else:
             base_obs_vector = self._get_vector_state(add_multiplier=add_multiplier)

        obs_dict = {agent: base_obs_vector.copy() for agent in self.agents}
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

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.map = copy.deepcopy(self.initMap)
        self._createItems()
        self.step_count = 0

        counter = Counter(self.task)
        self.taskCompletionStatus = [counter[element] if element in counter else 0 for element in TASKLIST]

        if self.human_role == "helpful":
            self.human_multiplier = 1.0
        elif self.human_role == "adversarial":
            self.human_multiplier = -1.0
        elif self.human_role == "random":
            self.human_multiplier = float(random.choice([1, -1]))
        else:
            print(f"Warning: Unknown human_role '{self.human_role}'. Defaulting to random.")
            self.human_multiplier = float(random.choice([1, -1]))
            
        standard_obs = self._get_obs()

        self.obs_history = {agent: [standard_obs[agent].copy()] for agent in self.agents}

        if self.frame_stack_size > 1:
             initial_stacked_obs = self._get_stacked_obs()
             return initial_stacked_obs, {}
        else:
             return standard_obs, {}
    
    def step(self, action):

        """
        Parameters
        ----------
        action: list
            action for each agent

        Returns
        -------
        obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        action = [action['human'], action['ai']] # some ugly hack to make the environment work with rllib.

        self.step_count += 1

        # Executing any action incurs a step penalty.
        # If the step count reaches 24, mark done as True and reset the counter.
        self.reward = self.rewardList["step penalty"]
        done = False

        info = {}
        info['cur_mac'] = action
        info['mac_done'] = [True] * self.n_agent
        info['collision'] = []

        all_action_done = False

        for agent in self.agent:

            agent.moved = False

        if self.debug:
            print("in overcooked primitive actions:", action)
            pass

        while not all_action_done:
            for idx, agent in enumerate(self.agent):
                agent_action = action[idx]
                if agent.moved:
                    continue
                agent.moved = True

                if agent_action < 4:
                    target_x = agent.x + DIRECTION[agent_action][0]
                    target_y = agent.y + DIRECTION[agent_action][1]
                    target_name = ITEMNAME[self.map[target_x][target_y]]

                    if target_name == "agent":
                        target_agent = self._findItem(target_x, target_y, target_name)
                        if not target_agent.moved:
                            agent.moved = False
                            target_agent_action = action[AGENTCOLOR.index(target_agent.color)]
                            if target_agent_action < 4:
                                new_target_agent_x = target_agent.x + DIRECTION[target_agent_action][0]
                                new_target_agent_y = target_agent.y + DIRECTION[target_agent_action][1]
                                if new_target_agent_x == agent.x and new_target_agent_y == agent.y:
                                    target_agent.move(new_target_agent_x, new_target_agent_y)
                                    agent.move(target_x, target_y)
                                    agent.moved = True
                                    target_agent.moved = True

                    # If the target is space, move the agent to the target position
                    elif  target_name == "space":
                        self.map[agent.x][agent.y] = ITEMIDX["space"]
                        agent.move(target_x, target_y)
                        self.map[target_x][target_y] = ITEMIDX["agent"]

                    # pickup and chop 
                    # If the agent doesn't have anything in hand
                    elif not agent.holding:
                        # If the target is a movable item, pick it up
                        if target_name in ["tomato", "lettuce", "plate", "onion"]:
                            item = self._findItem(target_x, target_y, target_name)
                            agent.pickup(item)
                            self.map[target_x][target_y] = ITEMIDX["counter"]

                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            # If the knife is holding a plate, pick it up
                            if isinstance(knife.holding, Plate):
                                item = knife.holding
                                knife.release()
                                agent.pickup(item)
                            # If the knife is holding food
                            elif isinstance(knife.holding, Food):
                                # If the food is chopped, pick it up
                                if knife.holding.chopped:
                                    item = knife.holding
                                    knife.release()
                                    agent.pickup(item)
                                # If the food is not chopped, chop it once
                                else:
                                    knife.holding.chop()
                                    self.reward += self.rewardList["subtask finished"]
                                    # If the food is chopped after chopping, check if it is part of the current task
                                    if knife.holding.chopped:
                                        for task in self.task:
                                            if knife.holding.rawName in task:
                                                # Reward for completing a mini task
                                                self.reward += self.rewardList["subtask finished"]
                    # put down
                    # If the agent is currently holding something
                    elif agent.holding:
                        # If the target is a counter, the agent will put down the item
                        if target_name == "counter":
                            if agent.holding.rawName in ["tomato", "lettuce", "onion", "plate"]:
                                # Change the counter to hold the item the agent was holding
                                self.map[target_x][target_y] = ITEMIDX[agent.holding.rawName]
                            # Reset the agent to not holding anything
                            agent.putdown(target_x, target_y)
                            self.reward += self.rewardList["metatask failed"]
                        # If the target is a plate
                        elif target_name == "plate":
                            # If the agent is holding food, check if it is chopped
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    # Reward for placing chopped food on a plate
                                    self.reward += self.rewardList["subtask finished"]
                                    plate = self._findItem(target_x, target_y, target_name)
                                    item = agent.holding
                                    # Put down the item and reset the agent
                                    agent.putdown(target_x, target_y)
                                    # Place the food on the plate
                                    plate.contain(item)
                            else:
                                self.reward += self.rewardList["metatask failed"]
                        # If the target is a knife
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            # If the knife is empty, place the item on the knife
                            if not knife.holding:
                                item = agent.holding
                                agent.putdown(target_x, target_y)
                                knife.hold(item)
                                if isinstance(item, Food):
                                    if item.chopped:
                                        # Penalty for placing chopped food back on the knife
                                        self.reward += self.rewardList["metatask failed"]
                                    else:
                                        # Reward for placing unchopped food on the knife
                                        self.reward += self.rewardList["subtask finished"]
                                else:
                                    self.reward += self.rewardList["metatask failed"]
                            # If the knife is holding food and the agent is holding a plate, place the food on the plate
                            elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
                                item = knife.holding
                                if item.chopped:
                                    self.reward += self.rewardList["subtask finished"]
                                    knife.release()
                                    agent.holding.contain(item)
                                else:
                                    # Penalty for placing unchopped food on a plate
                                    self.reward += self.rewardList["metatask failed"]
                            elif isinstance(knife.holding, Food) and not isinstance(agent.holding, Plate):
                                # Penalty for holding food while the knife is holding food
                                self.reward += self.rewardList["metatask failed"]
                            # If the knife is holding a plate and the agent is holding food, place the food on the plate
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
                                plate_item = knife.holding
                                food_item = agent.holding
                                if food_item.chopped:
                                    self.reward += self.rewardList["subtask finished"]
                                    knife.release()
                                    agent.pickup(plate_item)
                                    agent.holding.contain(food_item)
                                else:
                                    # Penalty for placing unchopped food on a plate
                                    self.reward += self.rewardList["metatask failed"]
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Plate):
                                # Penalty for holding a plate while the knife is holding a plate
                                self.reward += self.rewardList["metatask failed"]
                        # If the target is a delivery point
                        elif target_name == "delivery":
                            # If the agent is holding a plate, check if it contains food
                            if isinstance(agent.holding, Plate):
                                if agent.holding.containing:
                                    delivered_ingredients = set()
                                    valid_delivery = True
                                    for item in agent.holding.containing:
                                        if isinstance(item, Food):
                                            if item.chopped:
                                                delivered_ingredients.add(item.rawName)
                                            else:
                                                valid_delivery = False # Must be chopped
                                                break
                                        else:
                                            valid_delivery = False # Can only deliver food on plate
                                            break
                                    required_ingredients = {i for i, count in enumerate(self.taskCompletionStatus) if count > 0}
                                    if valid_delivery and delivered_ingredients == required_ingredients:
                                        # Correct delivery!
                                        self.reward += self.rewardList["correct delivery"]
                                        done = True # Assume delivering the main task finishes the episode

                                        # Refresh items (original logic seems okay here)
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        for f in food:
                                            f.refresh()
                                            self.map[f.x][f.y] = ITEMIDX[f.rawName]

                                    else:
                                        # Wrong delivery
                                        self.reward += self.rewardList["wrong delivery"]
                                        # Refresh items even on wrong delivery
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        if item.containing: # Check if plate wasn't empty
                                             food = item.containing
                                             for f in food:
                                                 f.refresh()
                                                 self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]

                                else: # Plate is empty
                                    self.reward += self.rewardList["wrong delivery"]
                                    plate = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.refresh()
                                    self.map[plate.x][plate.y] = ITEMIDX[plate.name]
                            else: # Holding something other than a plate
                                self.reward += self.rewardList["wrong delivery"]
                                held_item = agent.holding
                                agent.putdown(target_x, target_y)
                                held_item.refresh() # Refresh the wrongly delivered item
                                self.map[held_item.x][held_item.y] = ITEMIDX[held_item.rawName]
                        # If the target is food, the agent can only put down the item if it is holding a plate and the food is chopped
                        elif target_name in ["tomato", "lettuce", "onion"]:
                            item = self._findItem(target_x, target_y, target_name)
                            if item.chopped and isinstance(agent.holding, Plate):
                                self.reward += self.rewardList["subtask finished"]
                                agent.holding.contain(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]
                            elif not item.chopped and isinstance(agent.holding, Plate):
                                self.reward += self.rewardList["metatask failed"]
                            elif isinstance(agent.holding, Food):
                                self.reward += self.rewardList["metatask failed"]

            # End of the for loop for all agents' actions
            all_action_done = True

            # Check if any agent has not moved
            for agent in self.agent:
                if not agent.moved:
                    all_action_done = False

        terminateds = {"__all__": done or self.step_count >= 80} # Termination condition

        # Apply human multiplier to the base reward calculated during the step
        human_final_reward = self.reward * self.human_multiplier
        ai_final_reward = self.reward # AI gets the original reward

        rewards = {"human": human_final_reward, "ai": ai_final_reward}
        infos = {agent: info for agent in self.agents}
        truncateds = {'__all__': False}

        # Get next observation (standard or stacked)
        if self.frame_stack_size > 1:
             next_obs = self._get_stacked_obs()
        else:
             next_obs = self._get_obs() # Will include multiplier

        return next_obs, rewards, terminateds, truncateds, infos # Match gymnasium standard


    def render(self, mode='human'):
        return self.game.on_render()

    





