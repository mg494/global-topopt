#!/usr/bin/env python3

import numpy as np
from topopt.physical import Material
from topopt.mesh import Mesh, Displacement, Force
from fem.fem import FEModel,StructuralElement
import torch

from utils.post import plot_dof
import matplotlib.pyplot as plt

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
        # F = torch.Tensor(np.delete(Fvec,self.constrained_dofs))
        return K#,F

    def init_Kmat(self):
        self.Kmat = self._apply_bcs(self.model.K)#,self.Fvec

    def kill_elem(self,elem):
        self.Kmat = self._apply_bcs(self.model.kill_elem(elem))

    def solve(self,device="cuda"):
        K,F = self.Kmat.to(device),self.Fvec.to(device)
        u = torch.linalg.solve(K,F)
        uall = torch.zeros((self.ndofs,1)).to(device)
        uall[self.idx] = u
        return uall

    def strain_energy(self,deform:torch.Tensor,device="cuda"):
        klocs = torch.Tensor(np.load("edata.npy")).to(device)
        deform = deform[self.model.elem_to_dof_map]
        # deformT =deform[self.model.elem_to_dof_map].T
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
        return self.elem_state,strain_en

    def step(self,action):
        # substract selected element from stiffness matrix
        self.model.kill_elem(action)
        self.elem_state[action] = 0

        # solve for next state
        u = self.model.solve()
        new_strain_en = self.model.strain_energy(u)

        reward = (self.init_strain_energy/sum(new_strain_en))**2+(sum(self.elem_state/self.init_vol))**2
        self.count += 1
        if self.count > 50: done=True
        else: done = False
        return self.elem_state,reward,done #np.concatenate((new_x,new_strain_en))

class TopoAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self,state):
        return np.random.choice(self.action_size) # exploration only

    def remember(self,state,action,reward,next_state):
        pass

    def replay(self,batch_size):
        pass

# Mesh and Model init
ndiv = 20
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
# state = env.reset()
state_size = 2*ndiv**2 # env.observation_space.shape[0]
action_size = ndiv**2 # env.action_space.shape[0]
agent = TopoAgent(state_size,action_size)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state,reward,done = env.step(action)
        #print("killed elem",action,"reward",reward)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break
