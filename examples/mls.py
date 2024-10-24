#!/usr/bin/env python3

import numpy as np
from topopt.physical import Material
from topopt.mesh import Mesh, Displacement, Force
from fem.fem import FEModel,StructuralElement
import sys, logging, glob,os
from timeit import default_timer as timer

from utils.post import plot_dof
from topopt.env import TopoEnv
from topopt.rl import TopoAgent

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger('topopt')

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
episodes = 10000

env = TopoEnv(fem,support,load)
state_size = 2*ndiv**2
action_size = ndiv**2 
agent = TopoAgent(state_size,action_size)

fig,axs=plt.subplots(2,1)
colors = ["white", "grey","grey","blue"]
nodes = [0.0, 0.4, 0.6,1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

#outs = glob.glob("output/*.png")
#for f in outs: os.remove(f)

plot_rewards,plot_iters = [],[]
plot_s_reward,plot_v_reward= [],[]
plot_umax,plot_umin=[],[]
trials_fig,trials_axs=plt.subplots(1,1)
trials_fig.set_size_inches(12,12)

logger.info("starts training loop")
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    reward_series=[]
    start_time = timer()
    while True:
        # select action with actor critic rl
        action_one_hot,action_log_prob = agent.select_action(state)
        action = np.argmax(action_one_hot)

        next_state,reward,done = env.step(action)
        agent.update(state, action_log_prob, reward, next_state, done)

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
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Time: {end_time-start_time}, Iterations: {env.count}")
            #print("Elements taken:"+" ".join(list(map(str,env.elem_taken))))
            break

    plot_rewards.append(total_reward)
    plot_iters.append(env.count)
    plot_s_reward.append(env.strain_reward)
    plot_v_reward.append(env.vol_reward)
    plot_umax.append(env.umax)
    plot_umin.append(env.umin)

    metrics_fig,metrics_axs = plt.subplots(2,2)
    metrics_fig.set_size_inches(12,12)
    metrics_axs[0,0].plot(plot_rewards,label="total_reward")
    metrics_axs[0,0].legend(loc="upper left")
    metrics_axs[0,1].plot(plot_iters,label="no_of_iterations")
    metrics_axs[0,1].legend(loc="upper left")

    metrics_axs[1,0].plot(plot_s_reward,label="strain_reward")
    metrics_axs[1,0].plot(plot_v_reward,label="vol_reward")
    metrics_axs[1,0].legend(loc="upper left")

    metrics_axs[1,1].plot(plot_umax,label="umax")
    metrics_axs[1,1].plot(plot_umin,label="umin")
    metrics_axs[1,1].legend(loc="upper left")

        
    plt.savefig("output/metrics.png")
    plt.close(metrics_fig)

    trials_axs.plot(reward_series)

    plt.savefig("output/trials.png")
plt.close(trials_fig)