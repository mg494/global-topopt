import torch
import numpy as np

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
