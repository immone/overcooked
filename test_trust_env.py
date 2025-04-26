import numpy as np
from environment.Overcooked import Overcooked_multi

def test_human_role_and_frame_stacking():
    # Test parameters
    print("=== Testing human role and frame stacking ===")
    
    # Create environment with frame stacking and explicit human role
    env_params = {
        "grid_dim": [5, 5],
        "task": "tomato salad",
        "rewardList": {
            "subtask finished": 10,
            "metatask failed": -5,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1
        },
        "map_type": "A",
        "mode": "vector",
        "debug": True,
        "frame_stack_size": 3,  # Test with 3 frames
        "human_role": "helpful"  # Test with helpful role
    }
    
    env = Overcooked_multi(**env_params)
    
    # 1. Check observation space dimensions
    single_obs_size = len(env._get_obs()["ai"])  # Access the original observation
    expected_stacked_size = single_obs_size * env.frame_stack_size
    actual_size = env.observation_spaces["ai"].shape[0]
    
    print(f"Single observation size: {single_obs_size}")
    print(f"Expected stacked size: {expected_stacked_size}")
    print(f"Actual observation space size: {actual_size}")
    print(f"Correct size: {actual_size == expected_stacked_size}")
    
    # 2. Reset and check initial stacked observation
    obs, info = env.reset()
    print(f"Initial observation shape: {obs['ai'].shape}")
    
    # 3. Verify human role affects rewards
    print(f"Human role: {'helpful' if env.human_multiplier == 1 else 'adversarial'}")
    print(f"Human multiplier: {env.human_multiplier}")
    
    # 4. Take a few steps and check that frames are stacking
    actions = [
        {"human": 0, "ai": 3},  # Human right, AI up
        {"human": 1, "ai": 2},  # Human down, AI left
        {"human": 2, "ai": 1}   # Human left, AI down
    ]
    for i in range(3):
        print(f"Step {i+1}:")
        next_obs, rewards, done, _, _ = env.step(actions[i])
        
        print(f"Observation shape: {next_obs['ai'].shape}")
        print(f"Human reward: {rewards['human']}")
        print(f"AI reward: {rewards['ai']}")
        
        # Check if human reward matches role (negative for adversarial, positive for helpful)
        print(f"Human reward sign matches role: {rewards['human'] == -0.1 * env.human_multiplier}")
        
        # Verify we're actually stacking different frames (simple check)
        if i > 0 and env.frame_stack_size > 1:
            # Compare the first chunk of the stacked observation with the second chunk
            first_frame = next_obs['ai'][:single_obs_size]
            second_frame = next_obs['ai'][single_obs_size:2*single_obs_size]
            difference = np.sum(np.abs(first_frame - second_frame))
            print(f"Difference between stacked frames: {difference}")
            print(f"Frames are different: {difference > 0}")
    
    # 5. Test with a different role
    print("=== Testing with adversarial role ===")
    env_params["human_role"] = "adversarial"
    env = Overcooked_multi(**env_params)
    
    obs, info = env.reset()
    print(f"Human role: {'helpful' if env.human_multiplier == 1 else 'adversarial'}")
    print(f"Human multiplier: {env.human_multiplier}")
    
    action = {"human": 0, "ai": 0}
    next_obs, rewards, done, _, _ = env.step(action)
    print(f"Human reward: {rewards['human']}")
    print(f"AI reward: {rewards['ai']}")
    
    # Check if rewards are correctly inverted for adversarial role
    print(f"Human reward sign matches role: {rewards['human'] == -0.1 * env.human_multiplier}")

if __name__ == "__main__":
    test_human_role_and_frame_stacking()