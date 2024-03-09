#!/usr/bin/env python3

import numpy as np
from topopt.physical import Material
from topopt.mesh import Mesh, Displacement, Force
from fem.fem import FEModel,StructuralElement

from utils.post import plot_dof
import matplotlib.pyplot as plt

class TopoEnv():
    def __init__(self,fem,support,load):
        self.model = fem
        self.support = support
        self.load = load
        self.elem_state = []
        

    def _reward_function(self,state):
        return (self.init_strain_energy/sum())

    def _update_state(self,x):
        self.elem_state = x
        self.model.update_system_matrix(x)
        u = self.model.solve(self.support,load=self.load)
        strain_en = self.model.strain_energy(u)
        return x,strain_en

    def reset(self):
        x = np.ones(self.model.nelem,dtype=int)
        x,strain_en=self._update_state(x)
        self.init_vol = len(x)
        self.init_strain_energy = sum(strain_en)
        self.count =0
        return x #np.concatenate((x,strain_en))

    def step(self,action):
        x = self.elem_state - action*0.999
        new_x,new_strain_en=self._update_state(x)
        reward = (self.init_strain_energy/sum(new_strain_en))**2+(sum(new_x/self.init_vol))**2
        self.count += 1
        if self.count > 50: done=True 
        else: done = False
        return x,reward,done #np.concatenate((new_x,new_strain_en))

class TopoAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self,state):
        # exploration only
        idx = np.random.choice(self.action_size)
        action = np.zeros(self.action_size,dtype=int)
        action[idx] = int(1)
        return action

    def remember(self,state,action,reward,next_state):
        pass

    def replay(self,batch_size):
        pass
    
ndiv = 10

mesh = Mesh()
mesh.rect_mesh(ndiv)

mat = Material()
mat.set_structural_params(2.1e5,0.3)

support = Displacement(mesh)
support.add_by_plane([1,0],-1,0)

load = Force(mesh)
load.add_by_point((1,0),(0,-1))

fem = FEModel(mesh,mat,StructuralElement)
u = fem.solve(support,load=load)

#plot_dof(fem,u,0)
#plt.show()

episodes = 5

env = TopoEnv(fem,support,load)
state = env.reset()
state_size = 2*ndiv**2 # env.observation_space.shape[0]
action_size = ndiv**2 # env.action_space.shape[0]
agent = TopoAgent(state_size,action_size)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state,reward,done = env.step(action)
        state = next_state

        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break
