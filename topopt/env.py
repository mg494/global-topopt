import torch
import numpy as np

class TopoModel:
    def __init__(self,fem,support,load):

        # store basics
        self.model = fem
        self.nelem = fem.nelem
        self.ndofs = fem.ndofs
        
        # store element matrices
        self.klocs = torch.Tensor(np.load("edata.npy")).to("cuda")
        
        # store system of linear equations
        # Kmat holds the stiffness values of the fully filled domain
        self.Kmat = torch.Tensor(fem.K).to("cuda")
        self.Fvec = torch.zeros((self.ndofs,1))

        # store load data from loads object
        self.constrained_dofs = support.get_constrained_dofs(self.model.node_to_dof_map)
        self.Fvec = np.zeros(self.model.ndofs)
        self.Fvec[load.get_constrained_dofs(self.model.node_to_dof_map)] = load.get_constrained_values()
        self.Fvec = torch.Tensor(np.delete(self.Fvec,self.constrained_dofs))

        # init Kmat with bcs applied
        # Kidx holds all equation numbers of unconstrained dofs
        self.Kidx = list(range(self.ndofs))
        for num in self.constrained_dofs: self.Kidx.remove(num)

        # Ksub is the matrix where the stiffness terms of all deleted elements
        # are stored. It has the dimensions of the initial Kmat
        # and is substracted from Kmat
        self.Ksub=torch.zeros_like(self.Kmat).to("cuda")

        # The Tensor K holds the stiffness of the modified domain
        # the dimension is less than Ksub and Kmat and is used to solve
        # the system of linear equations
        self.K = self._apply_bcs(self.Kmat-self.Ksub)#[:,self.Kidx][self.Kidx,:].to("cuda")

        # index to broadcast u to uall
        self.idx = torch.ones((self.ndofs,1))
        for dof in self.constrained_dofs:
            self.idx[dof]=0
        self.idx =self.idx.to(torch.bool)
        self.idx =self.idx.to("cuda")
        
        # element to dof map
        self.e2dofmap = self.model.elem_to_dof_map
        
    def _apply_bcs(self,Kmat):
        return Kmat[:,self.Kidx][self.Kidx,:]
    
    def init_Kmat(self):
        self.Ksub = torch.zeros_like(self.Kmat).to("cuda")

    def kill_elem(self, elem):
        idx = self.e2dofmap[elem]
        self.Ksub[idx[:, None], idx] -= self.klocs[elem]
        self.K = self._apply_bcs(self.Kmat - self.Ksub)

    def solve(self,device="cuda"):
        K,F = self.K.to(device),self.Fvec.to(device)
        uall = torch.zeros((self.ndofs,1)).to(device)
        uall[self.idx] = torch.linalg.solve(K,F)
        return uall

    def strain_energy(self,deform:torch.Tensor,device="cuda"):
        deform = deform[self.model.elem_to_dof_map]
        tutu =torch.matmul(self.klocs,deform)
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
        self.elem_taken =[]

        # get initial stiffness matrix from fe model
        self.model.init_Kmat()

        # solve initial state
        u = self.model.solve()
        
        strain_en = self.model.strain_energy(u)
        
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
        self.umin=torch.min(u).cpu().numpy()
        self.umax=torch.max(u).cpu().numpy()

        new_strain_en = self.model.strain_energy(u)
        self.strain_reward = (1-sum(new_strain_en)/self.init_strain_energy)**2
        self.vol_reward = (1-sum(self.elem_state)/self.init_vol)**2
        reward = self.strain_reward+self.vol_reward

        if self.count > len(self.elem_state) or action in self.elem_taken: 
            done=True 
        else: 
            done = False

        self.count += 1
        self.elem_taken.append(action)
        return np.concatenate((self.elem_state,new_strain_en),axis=0),reward,done 
