#!/usr/bin/env python3

import numpy as np
from topopt.physical import Material
from topopt.mesh import Mesh, Displacement, Force
from fem.fem import FEModel,StructuralElement
import sys, logging

import torch
import torch.nn as nn
import torch.optim as optim

from utils.post import plot_dof
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger('topopt')

class TopoModel:
    def __init__(self,fem,support,load):
        self.model = fem
        self.nelem = fem.nelem
        self.ndofs = fem.ndofs

        self.Kmat = None
        self.Fvec = None

        # get fem object and apply boundary conditions
        # save updated dof maps
        # calc and save initial state as Kmat and deform vector
        # each kill elem call updates the Kmat
        # each solve call solves with the updated Kmats

        self.constrained_dofs = support.get_constrained_dofs(self.model.node_to_dof_map)
        self.Fvec = np.zeros(self.model.ndofs)
        self.Fvec[load.get_constrained_dofs(self.model.node_to_dof_map)] = load.get_constrained_values()
        self.Fvec = torch.Tensor(np.delete(self.Fvec,self.constrained_dofs))

        self.idx = torch.ones((self.ndofs,1))
        for dof in self.constrained_dofs:
            self.idx[dof]=0
        self.idx =self.idx.to(torch.bool)
        self.idx =self.idx.to("cuda")
        
    def _apply_bcs(self,Kmat):
        K = torch.Tensor(np.delete(np.delete(Kmat,self.constrained_dofs,axis=1),self.constrained_dofs,axis=0))
        return K
    
    def init_Kmat(self):
        self.Kmat = self._apply_bcs(self.model.K)#,self.Fvec

    def kill_elem(self,elem):
        self.Kmat = self._apply_bcs(self.model.kill_elem(elem))

    def solve(self,device="cuda"):
        K,F = self.Kmat.to(device),self.Fvec.to(device)
        uall = torch.zeros((self.ndofs,1)).to(device)
        uall[self.idx] = torch.linalg.solve(K,F)
        return uall

    def strain_energy(self,deform:torch.Tensor,device="cuda"):
        klocs = torch.Tensor(np.load("edata.npy")).to(device)
        deform = deform[self.model.elem_to_dof_map]
        tutu =torch.matmul(klocs,deform)
        return torch.matmul(deform.mT,tutu).cpu().numpy().flatten()

class TopoEnv():
    def __init__(self,fem,support,load):
        self.model = TopoModel(fem,support,load)
        self.elem_state = np.ones(self.model.nelem,dtype=int)
        self.init_vol = sum(self.elem_state)
        self.u = None

    def reset(self):
        # all elements in the design space are active
        self.elem_state = np.ones(self.model.nelem,dtype=int)

        # get initial stiffness matrix from fe model
        self.model.init_Kmat()

        # solve initial state
        self.u = self.model.solve()
        strain_en = self.model.strain_energy(self.u)
        
        # reset counter each episode
        self.count = 0
        self.init_strain_energy = sum(strain_en)

        return np.concatenate((self.elem_state,strain_en),axis=0)

    def step(self,action):
        # substract selected element from stiffness matrix
        self.model.kill_elem(action)
        self.elem_state[action] = 0

        # solve for next state
        u = self.model.solve()
        new_strain_en = self.model.strain_energy(u)

        reward = (self.init_strain_energy/sum(new_strain_en))**2+(sum(self.elem_state/self.init_vol))**2
        self.count += 1
        if self.count > len(self.elem_state): done=True 
        else: done = False
        return np.concatenate((self.elem_state,new_strain_en),axis=0),reward,done 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


class TopoAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_probs.shape[-1])
        return action_one_hot.squeeze(0).numpy(), action_dist.log_prob(action)
    
    def update(self, state, action_log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([1 - done])

        # Update Critic
        value = self.critic(state)
        next_value = self.critic(next_state)
        target_value = reward + self.gamma * next_value * done
        critic_loss = nn.MSELoss()(value, target_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        advantage = (target_value - value).detach()
        actor_loss = -action_log_prob * advantage

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# Mesh and Model init
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

# RL Training
episodes = 5000

env = TopoEnv(fem,support,load)
state_size = 2*ndiv**2
action_size = ndiv**2 
agent = TopoAgent(state_size,action_size)

fig,axs=plt.subplots(1,1)
colors = ["white", "grey","grey","blue"]
nodes = [0.0, 0.4, 0.6,1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

logger.info("starts training loop")
for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action_one_hot,action_log_prob = agent.select_action(state)
        action = np.argmax(action_one_hot)
        next_state,reward,done = env.step(action)
        agent.update(state, action_log_prob, reward, next_state, done)

        #print("killed elem",action,"reward",reward)
        state = next_state
        total_reward += reward
        if "plot" in sys.argv: 
            np.save("out.npy",env.elem_state)

        if "save" in sys.argv:
            axs.imshow(env.elem_state.reshape((ndiv,ndiv)),cmap=cmap,origin="lower")
            plt.savefig("output/elem_state_{}.png".format(episode))
        
        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break
