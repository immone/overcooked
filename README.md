# Macro-Action-Based Multi-Agent Reinforcement Learning

<p align="center">
    <img src="image/Overcooked_A.gif" width=260></img>
    <img src="image/Overcooked_B.gif" width=260></img>
    <img src="image/Overcooked_C.gif" width=260></img>
</p>

# Introduction
Adapted from [gym-cooking](https://github.com/rosewang2008/gym-cooking). 

Robots need to learn cooperating with each other to prepare a certain dish according to the recipe and deliver it to the `star' counter cell as soon as possible. The challenge is that the recipe is unknown to robots. Robots have to learn the correct procedure in terms of picking up raw vegetables, chopping, and merging in a plate before delivering.

## Installation

- To install all the dependencies:
```
pip install -U "ray[data,train,tune,serve]"
pip install pandas
pip install numpy
pip install matplotlib
pip install gymnasium
pip install pygame
pip install scipy
pip install tensorboard
pip install dm_tree
pip install torch
pip install pillow
pip install lz4
```
## Code structure
- `play.py`: a toy for mutual playing the env.
- `train_rllib.py`: training a marl model with rllib.
-  `run_trained.py`: test the trained model.
- `Agents.py`: the agents in the environment.
-  `./environment/Overcooked.py`: the main environment file.

- `./environment/items.py`: all the entities in the map(agent/food/plate/knife/delivery counter).

- `./environment/render`: resources and code for render.

### Manual control
```
python play.py
```
Adding the setting you want to play. Eg:
```
python play.py --task 6 --grid_dim 7 7 --map_type A
```
Enter the index of the action(primitive/macro) for each agent. The index of each action is listed in the file.  
Eg. When playing Overcooked_MA_V1, entering `1, 2 ,3`. Agent1 go to get tomato. Agent2 go to get lettuce. Agent3 go to get onion.

## Environment
| Map A | Map B | Map C |
|:---:|:---:|:---:|
|<img src="image/2_agent_A.png" width=260></img>  | <img src="image/2_agent_B.png" width=260></img> | <img src="image/2_agent_C.png" width=260></img> |
| <img src="image/3_agent_A.png" width=260></img>  | <img src="image/3_agent_B.png" width=260></img> | <img src="image/3_agent_C.png" width=260></img> |
| <img src="image/3_agent_A_9.png" width=260></img>  | <img src="image/3_agent_B_9.png" width=260></img> | <img src="image/3_agent_C_9.png" width=260></img> |


### Parameters
```
grid_dim(int, int): grid world size of the map
map_type(str): layout of the map
task(int): receipt agent cooks
mode(str): type of the observation
debug(bool): whehter print the debug information and render
```


- grid_dim
```
[5, 5]: the size of the map is 5X5 
[7, 7]: the size of the map is 7X7 
[9, 9]: the size of the map is 9X9
```

- map_type
```
A: map A
B: map B
C: map C
```

|<img src="image/3_agent_A.png" width=210></img> | <img src="image/obs1_1.png" width=90></img> | <img src="image/obs1_2.png" width=150></img> | <img src="image/obs1_3.png" width=210></img> |
|:---:|:---:|:---:|:---:|
| obs_radius: 0 | 1 | 2 | 3 |

- task
```
TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

task :
0 : tomato salad
1 : lettuce salad
2 : onion salad
3 : lettuce-tomato salad
4 : onion-tomato salad
5 : lettuce-onion salad
6 : lettuce-onion-tomato salad
```


|<img src="image/lettuce-tomato-salad.png" width=280></img> | <img src="image/lettuce-onion-tomato-salad.png" width=280></img> | 
|:---:|:---:|
| lettuce-tomato salad | lettuce-onion-tomato salad | 




- mode
```
vector: the observation is returned in vector 
image: the observation is returned in rgb array
```

## Observation
- vector  
```
obs = [tomato.x, tomato.y, tomato.status, lettuce.x, lettuce.y, lettuce.status, onion.x, onion.y, onion.status, plate-1.x, plate-1.y, plate-2.x, plate-2.y, knife-1.x, knife-1.y, knife-2.x, knife-2.y, delivery.x, delivery.y, agent1.x, agent1.y, agent2.x, agent2.y, (agent3.x, agent3.y), onehotTask]  

Agents only observe the positions and status of the entities within obs_radius. The items not observed are masked as 0 in the corresponding dims.
```

- image
```
if obs_radius > 0:
    height, width = 80 * (obs_radius * 2 + 1)
else:
    height, width = 80 * grid_dim
obs_size = [height, width, 3]
```

## Action
- Primitive-action  
right, down, left, up, stay


## Reward
- +10 for chopping a correct vegetable into pieces  
- +200 terminal reward for delivering the correct dish  
- −5 for delivering any wrong dish  
- −0.1 for every timestep  

## Termination
Env terminates when the correct dish is delivered.

## Extention
The values of reward can be changed in rewardList. Users can add new map of different layout by adding map in overcooked_V1.py. The new map is allowed to change the position of entities or delete any entities. Adding new entities is not supported.

## Citations
If you are using MacroMARL in your research, please cite the corresponding papers listed below:
```
@InProceedings{xiao_neurips_2022,
  author = "Xiao, Yuchen and Wei, Tan and Amato, Christopher",
  title = "Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning",
  booktitle = "Proceedings of the Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS)",
  year = "2022"
}
```

