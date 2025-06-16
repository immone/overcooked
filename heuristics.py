import numpy as np
from pathfinding import AStarPathfinder
from environment.items import Tomato, Lettuce, Onion, Plate, Knife, Food

class CooperativeHeuristic:
    """
    Cooperative heuristic that prioritizes efficient task completion.
    Follows the priority order from the proposal.
    """
    def __init__(self):
        """
        Initialize cooperative heuristic.
        
        TODO: Phase 3 - Extended Observations
        - Plate contents will be parsed from extended observation format
        - Agent holding status will be included in observations
        - Knife holding status will be parsed from observations
        """
        self.pathfinder = AStarPathfinder()
        self.last_action = None
        self.action_history = []
        
    def reset(self):
        """Reset internal state"""
        self.last_action = None
        self.action_history = []
        
    def get_action(self, env, agent_index):
        """
        Get cooperative action based on current state.
        
        Args:
            env: The environment instance
            agent_index: Index of this agent in env.agent list
            
        Priority order:
        1. Deliver completed salad
        2. Complete held plate
        3. Free hands if needed
        4. Fetch plate for assembly
        5. Chop ingredient
        6. Fetch ingredient
        7. Clear unnecessary item
        8. Default move
        """
        # Get the agent we're controlling
        if agent_index >= len(env.agent):
            return 4  # Stay if invalid index
            
        partner_agent = env.agent[agent_index]
        ai_agent = env.agent[1 - agent_index] if len(env.agent) > 1 else None
        
        # Get current position
        pos = (partner_agent.x, partner_agent.y)
        
        # Priority 1: Deliver completed salad
        if partner_agent.holding and isinstance(partner_agent.holding, Plate):
            plate = partner_agent.holding
            if plate.containing and self._is_correct_dish(plate, env):
                # Navigate to delivery and interact
                delivery_pos = self._find_nearest_item_pos("delivery", env)
                if delivery_pos:
                    if self._is_adjacent(pos, delivery_pos):
                        return self._get_interact_direction(pos, delivery_pos)
                    else:
                        return self._navigate_to(pos, delivery_pos, env)
        
        # Priority 2: Complete held plate
        if partner_agent.holding and isinstance(partner_agent.holding, Plate):
            plate = partner_agent.holding
            needed_items = self._get_needed_chopped_items(plate, env)
            if needed_items:
                # Find nearest chopped ingredient
                for item_type in needed_items:
                    chopped_pos = self._find_chopped_item_pos(item_type, env)
                    if chopped_pos:
                        if self._is_adjacent(pos, chopped_pos):
                            return self._get_interact_direction(pos, chopped_pos)
                        else:
                            return self._navigate_to(pos, chopped_pos, env)
        
        # Priority 3: Free hands if needed
        if partner_agent.holding and isinstance(partner_agent.holding, Plate):
            # Check if we need to chop something
            if self._need_to_chop_items(env):
                # Place plate on empty counter
                empty_counter = self._find_empty_counter(pos, env)
                if empty_counter:
                    if self._is_adjacent(pos, empty_counter):
                        return self._get_interact_direction(pos, empty_counter)
                    else:
                        return self._navigate_to(pos, empty_counter, env)
        
        # Priority 4: Fetch plate if needed
        if not partner_agent.holding and self._has_all_chopped_ingredients(env):
            plate_pos = self._find_nearest_item_pos("plate", env)
            if plate_pos:
                if self._is_adjacent(pos, plate_pos):
                    return self._get_interact_direction(pos, plate_pos)
                else:
                    return self._navigate_to(pos, plate_pos, env)
        
        # Priority 5: Chop ingredient
        if partner_agent.holding and isinstance(partner_agent.holding, Food):
            if not partner_agent.holding.chopped:
                # Find available knife
                knife_pos = self._find_available_knife(env)
                if knife_pos:
                    if self._is_adjacent(pos, knife_pos):
                        return self._get_interact_direction(pos, knife_pos)
                    else:
                        return self._navigate_to(pos, knife_pos, env)
        
        # Check if there's an item on a knife that needs chopping
        knife_with_food = self._find_knife_with_unchoped_food(env)
        if knife_with_food and not partner_agent.holding:
            if self._is_adjacent(pos, knife_with_food):
                return self._get_interact_direction(pos, knife_with_food)
            else:
                return self._navigate_to(pos, knife_with_food, env)
        
        # Priority 6: Fetch ingredient
        if not partner_agent.holding:
            needed_fresh = self._get_needed_fresh_items(env)
            for item_type in needed_fresh:
                item_pos = self._find_nearest_item_pos(item_type, env)
                if item_pos:
                    if self._is_adjacent(pos, item_pos):
                        return self._get_interact_direction(pos, item_pos)
                    else:
                        return self._navigate_to(pos, item_pos, env)
        
        # Priority 7: Clear unnecessary item
        if partner_agent.holding and not self._is_item_useful(partner_agent.holding, env):
            empty_counter = self._find_empty_counter(pos, env)
            if empty_counter:
                if self._is_adjacent(pos, empty_counter):
                    return self._get_interact_direction(pos, empty_counter)
                else:
                    return self._navigate_to(pos, empty_counter, env)
        
        # Priority 8: Default - move to center or stay
        center = (env.xlen // 2, env.ylen // 2)
        if pos != center and self._is_position_free(center, env):
            return self._navigate_to(pos, center, env)
        
        return 4  # Stay
    
    def _navigate_to(self, start, goal, env):
        """Navigate from start to goal using A*"""
        # Build obstacle map
        obstacles = self._build_obstacle_map(env)
        
        # Find path
        path = self.pathfinder.find_path(start, goal, obstacles, (env.xlen, env.ylen))
        
        if path and len(path) > 1:
            next_pos = path[1]
            # Convert to action
            dx = next_pos[0] - start[0]
            dy = next_pos[1] - start[1]
            
            if dx == 1:
                return 0  # Right
            elif dx == -1:
                return 2  # Left
            elif dy == 1:
                return 1  # Down
            elif dy == -1:
                return 3  # Up
        
        return 4  # Stay if no path
    
    def _build_obstacle_map(self, env):
        """Build set of obstacle positions"""
        obstacles = set()
        
        # Add counters and other static obstacles
        for x in range(env.xlen):
            for y in range(env.ylen):
                if env.map[x][y] == 1:  # Counter
                    obstacles.add((x, y))
        
        # Add other agents as obstacles
        for agent in env.agent:
            obstacles.add((agent.x, agent.y))
            
        return obstacles
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _get_interact_direction(self, from_pos, to_pos):
        """Get action to interact with adjacent position"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 1:
            return 0  # Right
        elif dx == -1:
            return 2  # Left
        elif dy == 1:
            return 1  # Down
        elif dy == -1:
            return 3  # Up
        else:
            return 4  # Stay (interact)
    
    def _find_nearest_item_pos(self, item_type, env):
        """Find position of nearest item of given type"""
        min_dist = float('inf')
        nearest_pos = None
        
        # Get partner position
        partner_pos = None
        for i, agent in enumerate(env.agent):
            if i == 0:  # Partner is always agent[0] (human)
                partner_pos = (agent.x, agent.y)
                break
                
        if not partner_pos:
            return None
        
        # Search for items
        if item_type in env.itemDic:
            for item in env.itemDic[item_type]:
                dist = abs(item.x - partner_pos[0]) + abs(item.y - partner_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = (item.x, item.y)
                    
        return nearest_pos
    
    def _find_empty_counter(self, pos, env):
        """Find nearest empty counter position"""
        min_dist = float('inf')
        nearest_pos = None
        
        for x in range(env.xlen):
            for y in range(env.ylen):
                if env.map[x][y] == 1:  # Counter
                    # Check if it's empty (no items on it)
                    occupied = False
                    for item_list in env.itemDic.values():
                        for item in item_list:
                            if item.x == x and item.y == y:
                                occupied = True
                                break
                        if occupied:
                            break
                            
                    if not occupied:
                        dist = abs(x - pos[0]) + abs(y - pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_pos = (x, y)
                            
        return nearest_pos
    
    def _is_position_free(self, pos, env):
        """Check if position is free (no obstacles or agents)"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= env.xlen or y < 0 or y >= env.ylen:
            return False
            
        # Check if it's a walkable tile
        if env.map[x][y] not in [0, 2]:  # Space or agent tile
            return False
            
        # Check if another agent is there
        for agent in env.agent:
            if agent.x == x and agent.y == y:
                return False
                
        return True
    
    def _is_correct_dish(self, plate, env):
        """Check if plate contains correct dish for current task"""
        if not plate.containing:
            return False
            
        # Build dish name from plate contents
        dish_name = ""
        food_types = {"tomato": False, "lettuce": False, "onion": False}
        
        for item in plate.containing:
            if isinstance(item, Tomato) and item.chopped:
                food_types["tomato"] = True
            elif isinstance(item, Lettuce) and item.chopped:
                food_types["lettuce"] = True
            elif isinstance(item, Onion) and item.chopped:
                food_types["onion"] = True
                
        # Build dish name in correct order
        if food_types["lettuce"]:
            dish_name += "lettuce-"
        if food_types["onion"]:
            dish_name += "onion-"
        if food_types["tomato"]:
            dish_name += "tomato-"
            
        if dish_name:
            dish_name = dish_name[:-1] + " salad"
            return dish_name == env.task
            
        return False
    
    def _get_needed_chopped_items(self, plate, env):
        """Get list of chopped items still needed for the recipe"""
        needed = []
        
        # Parse task to see what's needed
        task_items = set()
        if "tomato" in env.task:
            task_items.add("tomato")
        if "lettuce" in env.task:
            task_items.add("lettuce")
        if "onion" in env.task:
            task_items.add("onion")
            
        # Check what's already on the plate
        on_plate = set()
        if plate.containing:
            for item in plate.containing:
                if isinstance(item, Tomato) and item.chopped:
                    on_plate.add("tomato")
                elif isinstance(item, Lettuce) and item.chopped:
                    on_plate.add("lettuce")
                elif isinstance(item, Onion) and item.chopped:
                    on_plate.add("onion")
                    
        # Find what's still needed
        for item in task_items:
            if item not in on_plate:
                needed.append(item)
                
        return needed
    
    def _find_chopped_item_pos(self, item_type, env):
        """Find position of chopped item of given type"""
        # Check items on counters
        for item_list_name, item_list in env.itemDic.items():
            for item in item_list:
                if isinstance(item, Food) and item.chopped:
                    if (item_type == "tomato" and isinstance(item, Tomato)) or \
                       (item_type == "lettuce" and isinstance(item, Lettuce)) or \
                       (item_type == "onion" and isinstance(item, Onion)):
                        return (item.x, item.y)
                        
        # Check items on knives
        for knife in env.itemDic.get("knife", []):
            if knife.holding and isinstance(knife.holding, Food) and knife.holding.chopped:
                if (item_type == "tomato" and isinstance(knife.holding, Tomato)) or \
                   (item_type == "lettuce" and isinstance(knife.holding, Lettuce)) or \
                   (item_type == "onion" and isinstance(knife.holding, Onion)):
                    return (knife.x, knife.y)
                    
        return None
    
    def _need_to_chop_items(self, env):
        """Check if we still need to chop items for the recipe"""
        # Check all required ingredients
        required_items = []
        if "tomato" in env.task:
            required_items.append("tomato")
        if "lettuce" in env.task:
            required_items.append("lettuce")
        if "onion" in env.task:
            required_items.append("onion")
        
        # Check if all required items are chopped
        for item_type in required_items:
            chopped_found = False
            
            # Check counters and knives for chopped items
            for item in env.itemDic.get(item_type, []):
                if isinstance(item, Food) and item.chopped:
                    chopped_found = True
                    break
                    
            # Check knives
            for knife in env.itemDic.get("knife", []):
                if knife.holding and knife.holding.rawName == item_type and knife.holding.chopped:
                    chopped_found = True
                    break
                    
            if not chopped_found:
                return True
                
        return False
    
    def _has_all_chopped_ingredients(self, env):
        """Check if all needed ingredients are chopped and available"""
        required_items = []
        if "tomato" in env.task:
            required_items.append("tomato")
        if "lettuce" in env.task:
            required_items.append("lettuce")
        if "onion" in env.task:
            required_items.append("onion")
        
        if not required_items:
            return False
            
        # Check each required item
        for item_type in required_items:
            found = False
            
            # Check items on counters
            for item in env.itemDic.get(item_type, []):
                if isinstance(item, Food) and item.chopped:
                    found = True
                    break
                    
            # Check items on knives
            if not found:
                for knife in env.itemDic.get("knife", []):
                    if knife.holding and knife.holding.rawName == item_type and knife.holding.chopped:
                        found = True
                        break
                        
            if not found:
                return False
                
        return True
    
    def _find_available_knife(self, env):
        """Find position of available knife (empty or with unchoped food)"""
        for knife in env.itemDic.get("knife", []):
            if not knife.holding:
                return (knife.x, knife.y)
            elif isinstance(knife.holding, Food) and not knife.holding.chopped:
                return (knife.x, knife.y)
        return None
    
    def _find_knife_with_unchoped_food(self, env):
        """Find knife that has unchoped food on it"""
        for knife in env.itemDic.get("knife", []):
            if knife.holding and isinstance(knife.holding, Food) and not knife.holding.chopped:
                # Check if it's a needed ingredient
                if self._is_item_useful(knife.holding, env):
                    return (knife.x, knife.y)
        return None
    
    def _get_needed_fresh_items(self, env):
        """Get list of fresh items needed for recipe"""
        needed = []
        
        # Parse task to see what's needed
        if "tomato" in env.task:
            needed.append("tomato")
        if "lettuce" in env.task:
            needed.append("lettuce")
        if "onion" in env.task:
            needed.append("onion")
            
        return needed
    
    def _is_item_useful(self, item, env):
        """Check if item is useful for current recipe"""
        if isinstance(item, Tomato):
            return "tomato" in env.task
        elif isinstance(item, Lettuce):
            return "lettuce" in env.task
        elif isinstance(item, Onion):
            return "onion" in env.task
        elif isinstance(item, Plate):
            return True  # Plates are always useful
        return False


class SabotageHeuristic:
    """
    Sabotage heuristic that actively hinders progress.
    Cycles through different sabotage strategies.
    """
    def __init__(self):
        """
        Initialize sabotage heuristic.
        
        TODO: Phase 3 - Extended Observations
        - Will use extended observations to better predict AI agent intentions
        - Plate and knife contents will be available for spoiling strategies
        """
        self.pathfinder = AStarPathfinder()
        self.strategy_counter = 0
        self.strategy_duration = 15  # Steps per strategy
        self.current_strategy = None
        self.strategy_step = 0
        
        # Available strategies
        self.strategies = [
            "HogKnife",
            "BlockPath", 
            "MisplaceItem",
            "SpoilPlate",
            "FakeDelivery"
        ]
    def reset(self):
        """Reset internal state"""
        self.strategy_counter = 0
        self.current_strategy = None
        self.strategy_step = 0
        
    def get_action(self, env, agent_index):
        """
        Get sabotage action based on current strategy.
        
        Args:
            env: The environment instance
            agent_index: Index of this agent in env.agent list
        """
        # Update strategy if needed
        if self.strategy_step >= self.strategy_duration or self.current_strategy is None:
            self.current_strategy = self.strategies[self.strategy_counter % len(self.strategies)]
            self.strategy_counter += 1
            self.strategy_step = 0
            
        self.strategy_step += 1
        
        # Execute current strategy
        if self.current_strategy == "HogKnife":
            return self._hog_knife(env, agent_index)
        elif self.current_strategy == "BlockPath":
            return self._block_path(env, agent_index)
        elif self.current_strategy == "MisplaceItem":
            return self._misplace_item(env, agent_index)
        elif self.current_strategy == "SpoilPlate":
            return self._spoil_plate(env, agent_index)
        elif self.current_strategy == "FakeDelivery":
            return self._fake_delivery(env, agent_index)
        else:
            return 4  # Stay as default
    
    def _find_partner_agent(self, env, agent_index):
        """Get the saboteur agent"""
        if agent_index < len(env.agent):
            return env.agent[agent_index]
        return None
    
    def _find_ai_agent(self, env, agent_index):
        """Find the AI agent"""
        if len(env.agent) > 1:
            return env.agent[1 - agent_index]
        return None
    
    def _hog_knife(self, env, agent_index):
        """Strategy: Block access to knives"""
        partner = self._find_partner_agent(env, agent_index)
        if not partner:
            return 4
            
        pos = (partner.x, partner.y)
        
        # Find nearest knife
        knife_pos = self._find_nearest_knife(pos, env)
        if not knife_pos:
            return 4
            
        # If adjacent to knife, either place junk item or stay to block
        if self._is_adjacent(pos, knife_pos):
            knife = self._get_knife_at(knife_pos, env)
            if knife and not knife.holding and partner.holding:
                # Place whatever we're holding on the knife
                return self._get_interact_direction(pos, knife_pos)
            else:
                # Just stay to block
                return 4
        else:
            # Navigate to knife
            return self._navigate_to(pos, knife_pos, env)
    
    def _block_path(self, env, agent_index):
        """Strategy: Block AI agent's likely path"""
        partner = self._find_partner_agent(env, agent_index)
        ai = self._find_ai_agent(env, agent_index)
        if not partner or not ai:
            return 4
            
        # Predict where AI might be going
        ai_target = self._predict_ai_target(ai, env)
        if not ai_target:
            # Just follow AI agent
            return self._navigate_to((partner.x, partner.y), (ai.x, ai.y), env)
            
        # Find chokepoint between AI and target
        chokepoint = self._find_chokepoint((ai.x, ai.y), ai_target, env)
        if chokepoint:
            if (partner.x, partner.y) == chokepoint:
                return 4  # Stay to block
            else:
                return self._navigate_to((partner.x, partner.y), chokepoint, env)
                
        # Default: move towards AI
        return self._navigate_to((partner.x, partner.y), (ai.x, ai.y), env)
    
    def _misplace_item(self, env, agent_index):
        """Strategy: Move important items to inconvenient locations"""
        partner = self._find_partner_agent(env, agent_index)
        if not partner:
            return 4
            
        pos = (partner.x, partner.y)
        
        # If holding something, find far corner to drop it
        if partner.holding:
            far_counter = self._find_farthest_empty_counter(pos, env)
            if far_counter:
                if self._is_adjacent(pos, far_counter):
                    return self._get_interact_direction(pos, far_counter)
                else:
                    return self._navigate_to(pos, far_counter, env)
        else:
            # Find important item to pick up (chopped ingredients, plates with food)
            important_item_pos = self._find_important_item(env)
            if important_item_pos:
                if self._is_adjacent(pos, important_item_pos):
                    return self._get_interact_direction(pos, important_item_pos)
                else:
                    return self._navigate_to(pos, important_item_pos, env)
                    
        return 4
    
    def _spoil_plate(self, env, agent_index):
        """Strategy: Add wrong ingredients to plates"""
        partner = self._find_partner_agent(env, agent_index)
        if not partner:
            return 4
            
        pos = (partner.x, partner.y)
        
        # Find plate with some items
        partial_plate_pos = self._find_partial_plate(env)
        
        if partner.holding:
            # If holding wrong ingredient and near plate, add it
            if partial_plate_pos and self._is_adjacent(pos, partial_plate_pos):
                if isinstance(partner.holding, Food):
                    return self._get_interact_direction(pos, partial_plate_pos)
            else:
                # Navigate to plate
                if partial_plate_pos:
                    return self._navigate_to(pos, partial_plate_pos, env)
        else:
            # Pick up wrong ingredient
            wrong_item_pos = self._find_wrong_ingredient(env)
            if wrong_item_pos:
                if self._is_adjacent(pos, wrong_item_pos):
                    return self._get_interact_direction(pos, wrong_item_pos)
                else:
                    return self._navigate_to(pos, wrong_item_pos, env)
                    
        return 4
    
    def _fake_delivery(self, env, agent_index):
        """Strategy: Deliver wrong dishes"""
        partner = self._find_partner_agent(env, agent_index)
        if not partner:
            return 4
            
        pos = (partner.x, partner.y)
        
        if partner.holding and isinstance(partner.holding, Plate):
            # Navigate to delivery
            delivery_pos = self._find_delivery(env)
            if delivery_pos:
                if self._is_adjacent(pos, delivery_pos):
                    return self._get_interact_direction(pos, delivery_pos)
                else:
                    return self._navigate_to(pos, delivery_pos, env)
        else:
            # Pick up any plate
            plate_pos = self._find_any_plate(env)
            if plate_pos:
                if self._is_adjacent(pos, plate_pos):
                    return self._get_interact_direction(pos, plate_pos)
                else:
                    return self._navigate_to(pos, plate_pos, env)
                    
        return 4
    
    # Helper methods (similar structure to cooperative heuristic)
    def _navigate_to(self, start, goal, env):
        """Navigate from start to goal using A*"""
        obstacles = self._build_obstacle_map(env)
        path = self.pathfinder.find_path(start, goal, obstacles, (env.xlen, env.ylen))
        
        if path and len(path) > 1:
            next_pos = path[1]
            dx = next_pos[0] - start[0]
            dy = next_pos[1] - start[1]
            
            if dx == 1:
                return 0  # Right
            elif dx == -1:
                return 2  # Left  
            elif dy == 1:
                return 1  # Down
            elif dy == -1:
                return 3  # Up
                
        return 4  # Stay if no path
    
    def _build_obstacle_map(self, env):
        """Build set of obstacle positions"""
        obstacles = set()
        
        for x in range(env.xlen):
            for y in range(env.ylen):
                if env.map[x][y] == 1:  # Counter
                    obstacles.add((x, y))
                    
        for agent in env.agent:
            obstacles.add((agent.x, agent.y))
            
        return obstacles
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _get_interact_direction(self, from_pos, to_pos):
        """Get action to interact with adjacent position"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 1:
            return 0  # Right
        elif dx == -1:
            return 2  # Left
        elif dy == 1:
            return 1  # Down
        elif dy == -1:
            return 3  # Up
        else:
            return 4  # Stay
    
    def _find_nearest_knife(self, pos, env):
        """Find nearest knife position"""
        min_dist = float('inf')
        nearest_pos = None
        
        for knife in env.itemDic.get("knife", []):
            dist = abs(knife.x - pos[0]) + abs(knife.y - pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_pos = (knife.x, knife.y)
                
        return nearest_pos
    
    def _get_knife_at(self, pos, env):
        """Get knife object at position"""
        for knife in env.itemDic.get("knife", []):
            if knife.x == pos[0] and knife.y == pos[1]:
                return knife
        return None
    
    def _predict_ai_target(self, ai_agent, env):
        """Try to predict where AI agent is heading"""
        # Simple heuristic: if AI is holding something, might go to knife or delivery
        if ai_agent.holding:
            if isinstance(ai_agent.holding, Food) and not ai_agent.holding.chopped:
                # Probably going to knife
                return self._find_nearest_knife((ai_agent.x, ai_agent.y), env)
            elif isinstance(ai_agent.holding, Plate):
                # Might go to delivery or to get ingredients
                return self._find_delivery(env)
        else:
            # Might go to pick up ingredient
            for item_type in ["tomato", "lettuce", "onion"]:
                if item_type in env.task:
                    item_pos = self._find_nearest_item_pos(item_type, (ai_agent.x, ai_agent.y), env)
                    if item_pos:
                        return item_pos
                        
        return None
    
    def _find_chokepoint(self, start, goal, env):
        """Find a chokepoint position between start and goal"""
        # Simple implementation: find narrowest point on path
        obstacles = self._build_obstacle_map(env)
        path = self.pathfinder.find_path(start, goal, obstacles, (env.xlen, env.ylen))
        
        if not path or len(path) < 3:
            return None
            
        # Look for positions with few adjacent free spaces
        min_free = float('inf')
        chokepoint = None
        
        for pos in path[1:-1]:  # Skip start and goal
            free_count = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if new_pos not in obstacles and 0 <= new_pos[0] < env.xlen and 0 <= new_pos[1] < env.ylen:
                    free_count += 1
                    
            if free_count < min_free:
                min_free = free_count
                chokepoint = pos
                
        return chokepoint
    
    def _find_farthest_empty_counter(self, pos, env):
        """Find farthest empty counter from current position"""
        max_dist = -1
        farthest_pos = None
        
        for x in range(env.xlen):
            for y in range(env.ylen):
                if env.map[x][y] == 1:  # Counter
                    # Check if empty
                    occupied = False
                    for item_list in env.itemDic.values():
                        for item in item_list:
                            if item.x == x and item.y == y:
                                occupied = True
                                break
                        if occupied:
                            break
                            
                    if not occupied:
                        dist = abs(x - pos[0]) + abs(y - pos[1])
                        if dist > max_dist:
                            max_dist = dist
                            farthest_pos = (x, y)
                            
        return farthest_pos
    
    def _find_important_item(self, env):
        """Find important item to misplace (chopped ingredients, plates with food)"""
        # Prioritize chopped ingredients
        for item_type in ["tomato", "lettuce", "onion"]:
            for item in env.itemDic.get(item_type, []):
                if isinstance(item, Food) and item.chopped:
                    return (item.x, item.y)
                    
        # Then plates with food
        for plate in env.itemDic.get("plate", []):
            if plate.containing:
                return (plate.x, plate.y)
                
        return None
    
    def _find_partial_plate(self, env):
        """Find plate with some but not all ingredients"""
        for plate in env.itemDic.get("plate", []):
            if plate.containing and len(plate.containing) < 3:  # Not full
                return (plate.x, plate.y)
        return None
    
    def _find_wrong_ingredient(self, env):
        """Find ingredient not needed for current recipe"""
        not_needed = []
        
        if "tomato" not in env.task:
            not_needed.append("tomato")
        if "lettuce" not in env.task:
            not_needed.append("lettuce")
        if "onion" not in env.task:
            not_needed.append("onion")
            
        for item_type in not_needed:
            for item in env.itemDic.get(item_type, []):
                return (item.x, item.y)
                
        return None
    
    def _find_delivery(self, env):
        """Find delivery position"""
        for delivery in env.itemDic.get("delivery", []):
            return (delivery.x, delivery.y)
        return None
    
    def _find_any_plate(self, env):
        """Find any available plate"""
        for plate in env.itemDic.get("plate", []):
            return (plate.x, plate.y)
        return None
    
    def _find_nearest_item_pos(self, item_type, pos, env):
        """Find nearest item of given type to position"""
        min_dist = float('inf')
        nearest_pos = None
        
        if item_type in env.itemDic:
            for item in env.itemDic[item_type]:
                dist = abs(item.x - pos[0]) + abs(item.y - pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = (item.x, item.y)
                    
        return nearest_pos