import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents import PartnerRLM
from heuristics import CooperativeHeuristic, SabotageHeuristic
from pathfinding import AStarPathfinder, PathUtils
from environment.items import Tomato, Lettuce, Onion, Plate, Knife, Agent


class TestPartnerRLM:
    """Test suite for PartnerRLM class"""
    
    def test_quality_score_clipping(self):
        """Test that quality scores are properly clipped to [-1, 1]"""
        partner = PartnerRLM(
            observation_space=Mock(),
            action_space=Mock(sample=lambda: 0),
            inference_only=True
        )
        
        # Test normal range
        partner.set_quality_score(0.5)
        assert partner.quality_score == 0.5
        
        # Test clipping
        partner.set_quality_score(2.0)
        assert partner.quality_score == 1.0
        
        partner.set_quality_score(-2.0)
        assert partner.quality_score == -1.0
    
    def test_mixture_probabilities(self):
        """Test mixture probability calculations"""
        partner = PartnerRLM(
            observation_space=Mock(),
            action_space=Mock(),
            inference_only=True
        )
        
        # Test expert (q = 1.0)
        p_help, p_sab, p_noise = partner._compute_mixture_probabilities(1.0)
        assert abs(p_help - 0.95) < 1e-6
        assert abs(p_sab - 0.0) < 1e-6
        assert abs(p_noise - 0.05) < 1e-6
        assert abs(p_help + p_sab + p_noise - 1.0) < 1e-6
        
        # Test saboteur (q = -1.0)
        p_help, p_sab, p_noise = partner._compute_mixture_probabilities(-1.0)
        assert abs(p_help - 0.0) < 1e-6
        assert abs(p_sab - 0.95) < 1e-6
        assert abs(p_noise - 0.05) < 1e-6
        assert abs(p_help + p_sab + p_noise - 1.0) < 1e-6
        
        # Test novice (q = 0.0)
        p_help, p_sab, p_noise = partner._compute_mixture_probabilities(0.0)
        assert abs(p_help - 0.0) < 1e-6
        assert abs(p_sab - 0.0) < 1e-6
        assert abs(p_noise - 1.0) < 1e-6
        
        # Test intermediate positive (q = 0.4)
        p_help, p_sab, p_noise = partner._compute_mixture_probabilities(0.4)
        assert abs(p_help - 0.38) < 1e-6
        assert abs(p_sab - 0.0) < 1e-6
        assert abs(p_noise - 0.62) < 1e-6
        
        # Test intermediate negative (q = -0.3)
        p_help, p_sab, p_noise = partner._compute_mixture_probabilities(-0.3)
        assert abs(p_help - 0.0) < 1e-6
        assert abs(p_sab - 0.285) < 1e-6
        assert abs(p_noise - 0.715) < 1e-6
    
    def test_mixture_probability_properties(self):
        """Test mathematical properties of mixture probabilities"""
        partner = PartnerRLM(
            observation_space=Mock(),
            action_space=Mock(),
            inference_only=True
        )
        
        # Test that probabilities always sum to 1
        for q in np.linspace(-1, 1, 21):
            p_help, p_sab, p_noise = partner._compute_mixture_probabilities(q)
            assert abs(p_help + p_sab + p_noise - 1.0) < 1e-10
            
        # Test that all probabilities are non-negative
        for q in np.linspace(-1, 1, 21):
            p_help, p_sab, p_noise = partner._compute_mixture_probabilities(q)
            assert p_help >= 0
            assert p_sab >= 0
            assert p_noise >= 0
            
        # Test noise floor
        for q in np.linspace(-1, 1, 21):
            _, _, p_noise = partner._compute_mixture_probabilities(q)
            assert p_noise >= partner.epsilon


class TestCooperativeHeuristic:
    """Test suite for CooperativeHeuristic"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_env = Mock()
        self.mock_env.xlen = 5
        self.mock_env.ylen = 5
        self.mock_env.task = "tomato salad"
        self.mock_env.map = [[0 for _ in range(5)] for _ in range(5)]
        self.mock_env.agent = []
        self.mock_env.itemDic = {
            "tomato": [],
            "lettuce": [],
            "onion": [],
            "plate": [],
            "knife": [],
            "delivery": []
        }
        
    def test_initialization(self):
        """Test cooperative heuristic initialization"""
        heuristic = CooperativeHeuristic()  # Remove self.mock_env parameter
        assert isinstance(heuristic.pathfinder, AStarPathfinder)
        assert heuristic.last_action is None
        
    def test_deliver_completed_salad(self):
        """Test delivery priority when holding correct dish"""
        heuristic = CooperativeHeuristic()
        
        # Create partner agent holding correct plate
        partner = Mock()
        partner.color = "robot"
        partner.x = 2
        partner.y = 2
        
        # Create plate with correct ingredients
        plate = Mock(spec=Plate)
        plate.containing = [Mock(spec=Tomato, chopped=True, rawName="tomato")]
        partner.holding = plate
        
        # Create delivery point
        delivery = Mock()
        delivery.x = 4
        delivery.y = 4
        
        self.mock_env.agent = [partner]
        self.mock_env.itemDic["delivery"] = [delivery]
        
        # Mock the helper methods
        heuristic._is_correct_dish = Mock(return_value=True)
        heuristic._is_adjacent = Mock(return_value=False)
        heuristic._navigate_to = Mock(return_value=0)  # Move right
        
        state = {"env": self.mock_env}
        action = heuristic.get_action(state)
        
        assert action == 0  # Should navigate towards delivery
        heuristic._navigate_to.assert_called_once()
    
    def test_path_blocking_detection(self):
        """Test that cooperative heuristic avoids blocking paths"""
        heuristic = CooperativeHeuristic()
        
        # Set up narrow corridor
        self.mock_env.map = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        
        partner = Mock()
        partner.color = "robot"
        partner.x = 2
        partner.y = 2  # In the narrow passage
        partner.holding = None
        
        ai_agent = Mock()
        ai_agent.color = "blue"
        ai_agent.x = 1
        ai_agent.y = 2  # Trying to pass through
        
        self.mock_env.agent = [partner, ai_agent]
        
        state = {"env": self.mock_env}
        action = heuristic.get_action(state)
        
        # Should move out of the way (up or down)
        assert action in [1, 3]  # Down or Up


class TestSabotageHeuristic:
    """Test suite for SabotageHeuristic"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_env = Mock()
        self.mock_env.xlen = 5
        self.mock_env.ylen = 5
        self.mock_env.task = "tomato salad"
        self.mock_env.map = [[0 for _ in range(5)] for _ in range(5)]
        self.mock_env.agent = []
        self.mock_env.itemDic = {
            "tomato": [],
            "lettuce": [],
            "onion": [],
            "plate": [],
            "knife": [],
            "delivery": []
        }
        
    def test_strategy_cycling(self):
        """Test that sabotage strategies cycle correctly"""
        heuristic = SabotageHeuristic()
        
        # Create agents
        partner = Mock()
        partner.color = "robot"
        partner.x = 2
        partner.y = 2
        partner.holding = None
        
        ai_agent = Mock()
        ai_agent.color = "blue"
        ai_agent.x = 0
        ai_agent.y = 0
        
        self.mock_env.agent = [partner, ai_agent]
        
        # Track strategies used
        strategies_used = []
        
        # Mock strategy methods to track calls
        heuristic._hog_knife = Mock(return_value=4)
        heuristic._block_path = Mock(return_value=4)
        heuristic._misplace_item = Mock(return_value=4)
        heuristic._spoil_plate = Mock(return_value=4)
        heuristic._fake_delivery = Mock(return_value=4)
        
        state = {"env": self.mock_env}
        
        # Run for multiple strategy cycles
        for i in range(len(heuristic.strategies) * heuristic.strategy_duration + 5):
            action = heuristic.get_action(state)
            
        # Check that all strategies were called
        assert heuristic._hog_knife.called
        assert heuristic._block_path.called
        assert heuristic._misplace_item.called
        assert heuristic._spoil_plate.called
        assert heuristic._fake_delivery.called
    
    def test_knife_hogging(self):
        """Test knife hogging behavior"""
        heuristic = SabotageHeuristic()
        heuristic.current_strategy = "HogKnife"
        
        # Create agents
        partner = Mock()
        partner.color = "robot"
        partner.x = 2
        partner.y = 2
        partner.holding = Mock()  # Holding something
        
        # Create knife
        knife = Mock()
        knife.x = 3
        knife.y = 2
        knife.holding = None
        
        self.mock_env.agent = [partner]
        self.mock_env.itemDic["knife"] = [knife]
        
        # Mock helper methods
        heuristic._find_nearest_knife = Mock(return_value=(3, 2))
        heuristic._is_adjacent = Mock(return_value=True)
        heuristic._get_knife_at = Mock(return_value=knife)
        heuristic._get_interact_direction = Mock(return_value=0)
        
        state = {"env": self.mock_env}
        action = heuristic._hog_knife(self.mock_env)
        
        # Should interact with knife to place item
        assert action == 0


class TestAStarPathfinder:
    """Test suite for A* pathfinding"""
    
    def test_simple_path(self):
        """Test pathfinding in open area"""
        pathfinder = AStarPathfinder()
        
        start = (0, 0)
        goal = (4, 4)
        obstacles = set()
        grid_size = (5, 5)
        
        path = pathfinder.find_path(start, goal, obstacles, grid_size)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) == 9  # Manhattan distance + 1
    
    def test_path_with_obstacles(self):
        """Test pathfinding around obstacles"""
        pathfinder = AStarPathfinder()
        
        start = (0, 0)
        goal = (4, 0)
        # Create wall blocking direct path
        obstacles = {(1, 0), (2, 0), (3, 0)}
        grid_size = (5, 5)
        
        path = pathfinder.find_path(start, goal, obstacles, grid_size)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        # Check that path goes around obstacles
        for pos in path:
            assert pos not in obstacles
    
    def test_no_path_exists(self):
        """Test when no path exists"""
        pathfinder = AStarPathfinder()
        
        start = (0, 0)
        goal = (4, 4)
        # Surround goal with obstacles
        obstacles = {(3, 4), (4, 3), (3, 3)}
        # Add more obstacles to completely block
        for x in range(5):
            obstacles.add((x, 2))
        grid_size = (5, 5)
        
        path = pathfinder.find_path(start, goal, obstacles, grid_size)
        
        assert path is None
    
    def test_path_caching(self):
        """Test that paths are cached correctly"""
        pathfinder = AStarPathfinder()
        
        start = (0, 0)
        goal = (4, 4)
        obstacles = set()
        grid_size = (5, 5)
        
        # First call
        path1 = pathfinder.find_path(start, goal, obstacles, grid_size)
        
        # Second call should use cache
        path2 = pathfinder.find_path(start, goal, obstacles, grid_size)
        
        assert path1 == path2
        assert len(pathfinder.path_cache) == 1
    
    def test_manhattan_heuristic(self):
        """Test Manhattan distance calculation"""
        pathfinder = AStarPathfinder()
        
        assert pathfinder._heuristic((0, 0), (3, 4)) == 7
        assert pathfinder._heuristic((2, 2), (2, 2)) == 0
        assert pathfinder._heuristic((5, 5), (0, 0)) == 10


class TestPathUtils:
    """Test suite for path utilities"""
    
    def test_path_blocked_detection(self):
        """Test detection of blocked paths"""
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        obstacles = {(2, 0)}
        
        assert PathUtils.is_path_blocked(path, obstacles) == True
        
        obstacles = {(4, 0)}
        assert PathUtils.is_path_blocked(path, obstacles) == False
    
    def test_alternative_goal_finding(self):
        """Test finding alternative goals when original is blocked"""
        original_goal = (2, 2)
        obstacles = {(2, 2)}
        grid_size = (5, 5)
        
        alt_goal = PathUtils.find_alternative_goal(original_goal, obstacles, grid_size)
        
        assert alt_goal is not None
        assert alt_goal != original_goal
        assert alt_goal not in obstacles
        # Should be adjacent to original goal
        assert abs(alt_goal[0] - original_goal[0]) <= 1
        assert abs(alt_goal[1] - original_goal[1]) <= 1
    
    def test_path_smoothing(self):
        """Test path smoothing removes unnecessary waypoints"""
        # Zigzag path that can be straightened
        path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
        obstacles = set()
        
        smoothed = PathUtils.smooth_path(path, obstacles)
        
        assert len(smoothed) < len(path)
        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]
    
    def test_path_length_calculation(self):
        """Test path length calculation"""
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert PathUtils.get_path_length(path) == 3
        
        assert PathUtils.get_path_length([]) == 0
        assert PathUtils.get_path_length([(0, 0)]) == 0


class TestIntegration:
    """Integration tests for the complete partner system"""
    
    def test_partner_quality_behavior(self):
        """Test that partner behaves according to quality score"""
        # This would require a more complete mock environment
        # For now, we just test the basic flow
        
        partner = PartnerRLM(
            observation_space=Mock(),
            action_space=Mock(sample=lambda: np.random.randint(5)),
            inference_only=True
        )
        
        mock_env = Mock()
        
        # Test expert behavior (mostly cooperative)
        partner.set_quality_score(0.9)
        actions = []
        
        # We can't fully test without proper environment setup,
        # but we can verify the structure is correct
        assert hasattr(partner, 'cooperative_heuristic')
        assert hasattr(partner, 'sabotage_heuristic')
        assert partner.quality_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])