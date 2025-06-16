import heapq
from typing import List, Tuple, Set, Optional

class AStarPathfinder:
    """
    A* pathfinding implementation for grid-based navigation.
    Uses Manhattan distance as heuristic.
    """
    
    def __init__(self):
        self.path_cache = {}  # Cache for repeated queries
        self.cache_size = 100  # Maximum cache size
        
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  obstacles: Set[Tuple[int, int]], grid_size: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal avoiding obstacles.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: Set of obstacle positions
            grid_size: Size of the grid (width, height)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        # Check cache first
        cache_key = (start, goal, frozenset(obstacles))
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Check if start or goal are obstacles
        if start in obstacles or goal in obstacles:
            return None
            
        # Check if start equals goal
        if start == goal:
            return [start]
        
        # Initialize A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        visited = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check if we reached the goal
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                self._update_cache(cache_key, path)
                return path
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor in obstacles or neighbor in visited:
                    continue
                    
                tentative_g = g_score[current] + 1  # Cost is 1 for each step
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        # No path found
        self._update_cache(cache_key, None)
        return None
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int], grid_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        x, y = pos
        width, height = grid_size
        
        # Four cardinal directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < width and 0 <= new_y < height:
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
            
        path.reverse()
        return path
    
    def _update_cache(self, key, path):
        """Update path cache with size limit"""
        if len(self.path_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]
            
        self.path_cache[key] = path
    
    def clear_cache(self):
        """Clear the path cache"""
        self.path_cache.clear()


class PathUtils:
    """Utility functions for path analysis and manipulation"""
    
    @staticmethod
    def is_path_blocked(path: List[Tuple[int, int]], obstacles: Set[Tuple[int, int]]) -> bool:
        """Check if any position in path is blocked by obstacles"""
        if not path:
            return False
            
        for pos in path:
            if pos in obstacles:
                return True
                
        return False
    
    @staticmethod
    def find_alternative_goal(original_goal: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                            grid_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest accessible position to original goal if it's blocked"""
        if original_goal not in obstacles:
            return original_goal
            
        x, y = original_goal
        width, height = grid_size
        
        # Search in expanding rings around original goal
        for radius in range(1, max(width, height)):
            positions = []
            
            # Generate positions at this radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only edge positions
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < width and 0 <= new_y < height:
                            if (new_x, new_y) not in obstacles:
                                positions.append((new_x, new_y))
            
            # Return first valid position found
            if positions:
                return positions[0]
                
        return None
    
    @staticmethod
    def smooth_path(path: List[Tuple[int, int]], obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Remove unnecessary waypoints from path (simple line-of-sight smoothing)"""
        if not path or len(path) < 3:
            return path
            
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Look ahead to find furthest reachable position
            j = len(path) - 1
            while j > i + 1:
                if PathUtils._has_line_of_sight(path[i], path[j], obstacles):
                    break
                j -= 1
                
            smoothed.append(path[j])
            i = j
            
        return smoothed
    
    @staticmethod
    def _has_line_of_sight(start: Tuple[int, int], end: Tuple[int, int], 
                          obstacles: Set[Tuple[int, int]]) -> bool:
        """Check if there's a clear line of sight between two positions"""
        # Use Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if (x, y) in obstacles and (x, y) != start:
                return False
                
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return True
    
    @staticmethod
    def get_path_length(path: List[Tuple[int, int]]) -> int:
        """Get the length of a path"""
        if not path:
            return 0
        return len(path) - 1  # Number of steps