#!/usr/bin/env python3

import numpy as np
from topopt.physical import Material
from topopt.mesh import Mesh, Displacement, Force
from fem.fem import FEModel,StructuralElement
import sys, logging, glob,os
from timeit import default_timer as timer

from utils.post import plot_dof
from topopt.env import TopoEnv
from topopt.rl import A2CAgent,DQNAgent

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('topopt')
tb_writer= SummaryWriter()

# Mesh and Model init
ndiv = 20
mesh = Mesh()
mesh.rect_mesh(ndiv)

mat = Material()
mat.set_structural_params(2.1e5,0.3)

support = Displacement(mesh)
support.add_by_plane([1,0],-1,0)

load = Force(mesh)
load.add_by_point((1,0),(0,-100))

fem = FEModel(mesh,mat,StructuralElement)

# RL Training
episodes = 50000
batch_size = 40


env = TopoEnv(fem,support,load)
state_size = 2*ndiv**2
action_size = ndiv**2 
agent = DQNAgent(state_size,action_size)

fig,axs=plt.subplots(2,1)
colors = ["white", "grey","grey","blue"]
nodes = [0.0, 0.4, 0.6,1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

logger.info("starts training loop")
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    reward_series=[]
    start_time = timer()
    while True:
        # # select action with actor critic rl
        # action_one_hot,action_log_prob = agent.select_action(state)
        # action = np.argmax(action_one_hot)

        # next_state,reward,done = env.step(action)
        # agent.update(state, action_log_prob, reward, next_state, done)

        # dqn agent
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        reward_series.append(total_reward)

        if "plot" in sys.argv: 
            np.save("out.npy",env.elem_state)

        if "save" in sys.argv:
            axs[0].imshow(env.elem_state.reshape((ndiv,ndiv)),cmap=cmap,origin="lower")
            plt.savefig("output/elem_state_{}.png".format(episode))
        
        if done:
            end_time = timer()
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Time: {end_time-start_time}, Iterations: {env.count}, Exploration rate: {agent.epsilon}")

            break

    agent.replay(batch_size)
    
    tb_writer.add_scalar("Total Reward",total_reward,episode)
    tb_writer.add_scalar("Number of Iterations",env.count,episode)
    tb_writer.add_scalar("Strain Reward",env.strain_reward,episode)
    tb_writer.add_scalar("Volume Reaward",env.vol_reward,episode)
    tb_writer.add_scalar("Deformation Minimum",env.umin,episode)

tb_writer.close()
