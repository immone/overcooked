#!/usr/bin/env python3
import os
import time
import pygame
import numpy as np
from environment.Overcooked import Overcooked_multi

# Environment parameters
reward_config = {
    "subtask finished": 10,
    "metatask failed": -5,
    "correct delivery": 200,
    "wrong delivery": -50,
    "step penalty": -0.1
}

def debug_environment(map_type, task_name):
    print(f"===== Testing Map Type: {map_type}, Task: {task_name} =====")
    
    # Create environment
    env_params = {
        "grid_dim": [5, 5],
        "task": task_name,
        "rewardList": reward_config,
        "map_type": map_type,
        "mode": "vector",
        "debug": True,
        "human_role": "helpful",
        "frame_stack_size": 1
    }
    
    try:
        # Initialize environment
        env = Overcooked_multi(**env_params)
        
        # Debug map initialization
        print(f"Map dimensions: {env.xlen}x{env.ylen}")
        print("Map layout:")
        for row in env.map:
            print(row)
        
        # Debug item creation
        print("\nItems created:")
        print(f"Tomatoes: {len(env.tomato)}")
        print(f"Lettuce: {len(env.lettuce)}")
        print(f"Onions: {len(env.onion)}")
        print(f"Plates: {len(env.plate)}")
        print(f"Knives: {len(env.knife)}")
        print(f"Delivery: {len(env.delivery)}")
        
        # Print item locations
        print("")
        print("Item locations:")
        if env.tomato:
            print(f"Tomato locations: {[(t.x, t.y) for t in env.tomato]}")
        if env.lettuce:
            print(f"Lettuce locations: {[(l.x, l.y) for l in env.lettuce]}")
        if env.onion:
            print(f"Onion locations: {[(o.x, o.y) for o in env.onion]}")
        
        # Show current task requirements
        print(f"\nTask requirements: {env.task}")
        print(f"One-hot task encoding: {env.oneHotTask}")
        
        # Initialize pygame rendering
        try:
            env.game.on_init()
            print("\nPygame initialized successfully")
            
            # Debug graphics path
            graphics_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment/render/graphics'))
            print(f"Graphics directory: {graphics_dir}")
            
            # Reset environment
            obs, _ = env.reset()
            print("Environment reset successfully")
            
            # Debug rendering
            print("Rendering environment...")
            env.render()
            
            # Take a few random steps to see how rendering changes
            for i in range(10):
                actions = {
                    "human": env.action_spaces["human"].sample(),
                    "ai": env.action_spaces["ai"].sample()
                }
                next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
                print(f"\nStep {i+1} - Actions: human={actions['human']}, ai={actions['ai']}")
                print(f"Step {i+1} - Rewards: {rewards}")
                
                # Render after each step
                env.render()
                time.sleep(0.5)
            
            # Keep window open to observe rendering
            print("\nRender window open. Press Ctrl+C to exit...")
            try:
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Exiting environment test...")
                
        except Exception as e:
            print(f"Error during rendering: {e}")
        
    except Exception as e:
        print(f"Error initializing environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test multiple configurations
    map_types = ["A", "B", "C"]
    tasks = ["tomato salad", "lettuce salad", "onion salad", "lettuce-onion-tomato salad"]
    
    # Let user choose or test all combinations
    print("Choose a test option:")
    print("1. Test all map types and tasks")
    print("2. Test specific map type and task")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Test all combinations
        for map_type in map_types:
            for task in tasks:
                debug_environment(map_type, task)
                input("Press Enter to continue to next configuration...")
    else:
        # Let user choose specific configuration
        print("\nAvailable map types:")
        for i, map_type in enumerate(map_types):
            print(f"{i+1}. {map_type}")
        map_choice = int(input("Enter map type number: ")) - 1
        
        print("\nAvailable tasks:")
        for i, task in enumerate(tasks):
            print(f"{i+1}. {task}")
        task_choice = int(input("Enter task number: ")) - 1
        
        # Run the debug with selected configuration
        debug_environment(map_types[map_choice], tasks[task_choice])