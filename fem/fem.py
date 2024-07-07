import numpy as np
import logging
from topopt.mesh import Displacement,Force
import torch, sys,os

logger = logging.getLogger('topopt')


class FEModel:
    def __init__(self,mesh,mat,ElementTypeClass):
        self.elements = mesh.elements
        self.nelem = len(mesh.elements[:,0])
        self.nodes = mesh.nodal_coords
        self.dofs_per_node = ElementTypeClass.dofs_per_node
        self.ndofs = mesh.nnodes * self.dofs_per_node

        self.node_to_dof_map = self._make_node_to_dof_map(self.nodes,self.dofs_per_node)
        self.elem_to_dof_map = self._make_elem_to_dof_map(self.elements,self.node_to_dof_map)
        # 3d matrix to save local system matrices for post processing
        os.remove("edata.npy")
        self.klocs = None

        # instantiate element type for each unique 
        # element and material combination
        et_mat = self.elements[:,1:3]
        et_mat = np.unique(et_mat,axis=0)

        # Array of element types used in mesh
        self.ets = {et:ElementTypeClass(et,mat) for et,_ in et_mat}
        for e in range(1,len(self.ets)+1):
            logger.info("Element Type {}, {}".format(e,self.ets[e].__str__()))

        # assemble system matrix
        self._assemble()

    def _make_node_to_dof_map(self,nodes,ndofs):
        nnodes = nodes.shape[0]
        dofs = np.arange(nnodes*ndofs)
        return dofs.reshape((nnodes,ndofs))

    def _make_elem_to_dof_map(self,elements,node_to_dof_map):
        elem_to_dof_map = np.zeros((elements.shape[0],4*self.dofs_per_node),dtype=int)
        for idx,el in enumerate(elements):
            nodes_on_elem = el[3:]
            dofs_on_elem = node_to_dof_map[nodes_on_elem].flatten()
            elem_to_dof_map[idx,:] = dofs_on_elem
        return elem_to_dof_map

    def _assemble(self,x=None):
        self._K = np.zeros((self.ndofs,self.ndofs))
        kls = np.ndarray((self.nelem,8,8))
        for idx,el in enumerate(self.elements):
            el_no = el[0]
            et_no = el[1]
            nodes_on_elem = el[3:]
            nodal_coords = self.nodes[nodes_on_elem]
            dofs = self.elem_to_dof_map[el_no]
            kl = self.ets[et_no].kloc(nodal_coords)
            kls[idx,:,:] = kl
            rows = np.tile(dofs.reshape(len(dofs),1),len(dofs))
            cols = np.tile(dofs.reshape(1,len(dofs)),(len(dofs),1))
            if not x is None:
                self._K[rows,cols] += kl*x[idx]
            else: 
                self._K[rows,cols] += kl
        np.save("edata",kls)
        if "edata.npy" in os.listdir("./"): logger.info("saved edata to disk")

    def kill_elem(self,elem,fact=0.999):
        el = self.elements[elem]
        el_no = el[0]
        et_no = el[1]
        nodes_on_elem = el[3:]
        nodal_coords = self.nodes[nodes_on_elem]
        dofs = self.elem_to_dof_map[el_no]
        kl = self.ets[et_no].kloc(nodal_coords)
        rows = np.tile(dofs.reshape(len(dofs),1),len(dofs))
        cols = np.tile(dofs.reshape(1,len(dofs)),(len(dofs),1))
        self._K[rows,cols] -= fact*kl
        return self._K
    
    @property
    def K(self):
        return self._K
    
    def update_system_matrix(self,x):
        self._assemble(x)

    def solve(self,support,load=None):
        neq = self._K.shape[0]
        F = np.zeros(neq)
        u = np.zeros(neq)

        dofs = np.arange(neq)
        constrained_dofs = support.get_constrained_dofs(self.node_to_dof_map)
        unconstrained_dofs = np.delete(dofs,constrained_dofs)
        constrained_u = support.get_constrained_values()
        u[constrained_dofs] = constrained_u

        if not load is None:
            loaded_dofs = load.get_constrained_dofs(self.node_to_dof_map)
            loaded_values = load.get_constrained_values()
        else:
            loaded_dofs = []
            loaded_values = []

        F[loaded_dofs] = loaded_values
        
        # separate rows and cols for known dofs from those for unknown dofs
        Kk = self._K[np.ix_(constrained_dofs,constrained_dofs)]
        Kku = self._K[np.ix_(unconstrained_dofs,constrained_dofs)]
        Ku = np.delete(self._K,constrained_dofs,axis=0)
        Ku = np.delete(Ku,constrained_dofs,axis=1)
        
        Fu = np.delete(F,constrained_dofs)
        uk = u[constrained_dofs]
        uu = np.delete(u,constrained_dofs)

        # solve
        uu = np.linalg.solve(Ku,Fu-np.dot(Kku,uk))

        # reassemble system variable
        u = np.concatenate((uu,uk))
        dofs_unsorted = np.concatenate((unconstrained_dofs,constrained_dofs))
        idx_sort_u = np.argsort(dofs_unsorted)

        return u[idx_sort_u]

class ElementTypeBase:
    def __init__(self,et):

        ElementType = _element_types[et]
        self.ele = ElementType(self.dofs_per_node)
        self.ndofs = self.ele.nnodes * self.dofs_per_node   # dofs per node is inherited

        self.nnodes = self.ele.nnodes # Number of nodes
        self.ngpoints = self.ele.ngpoints # Number of gauss points
        self.gpoints = self.ele.gpoints
        self.gweights = self.ele.gweights
        self.dhdx = self.ele.dhdx

    # dummy methods, get overwritten with inheritance
    
    def jacobian(self,dhdx,coord):
        return dhdx.dot(coord)

    def det_jacobian(self,dhdx,coord):
        jaco = self.jacobian(dhdx,coord)
        det = np.linalg.det(jaco)
        return det

    def inv_jacobian(self,dhdx,coord):
        jacobian = self.jacobian(dhdx,coord)
        return np.linalg.inv(jacobian)

    def kloc(self,coord):
        kl = np.zeros([self.ndofs, self.ndofs])
        XP = self.gpoints
        XW = self.gweights
        for i in range(0, self.ngpoints):
            ri = XP[i, 0]
            si = XP[i, 1]
            alf = XW[i]
            dhdx = self.dhdx(ri,si)
            ddet = self.det_jacobian(dhdx,coord)
            B = self.B(ri, si, coord)
            kl = kl + np.dot(np.dot(B.T,self.C), B)*alf*ddet
        return kl

class StructuralElement(ElementTypeBase):
    dofs_per_node = 2
    def __init__(self,et,mat):
        self._et = et
        self.youngs = mat.youngs
        self.nu = mat.nu
        super().__init__(et)

    def __str__(self):
        return "Structural "+_element_types[self._et].__str__()

    @property
    def C(self):
        enu = self.youngs/(1 - self.nu**2)
        mnu = (1 - self.nu)/2
        return np.array([[enu,self.nu*enu,0],
                         [self.nu*enu,enu,0],
                         [0,0,enu*mnu]])

    def B(self,r,s,coord):
        B = np.zeros((3, self.ndofs))
        dhdx = self.dhdx(r,s)
        jaco_inv = self.inv_jacobian(dhdx, coord)
        dhdx = np.dot(jaco_inv, dhdx)
        B[0, ::2] = dhdx[0, :]
        B[1, 1::2] = dhdx[1, :]
        B[2, ::2] = dhdx[1, :]
        B[2, 1::2] = dhdx[0, :]
        return B

class ThermalElement(ElementTypeBase):
    dofs_per_node = 1
    def __init__(self,et,mat):
        self._et = et
        super().__init__(et)
        self.k = mat.k
    @property
    def C(self):
        return self.k*np.eye(2)
    def __str__(self):
        return "Thermal "+_element_types[self._et].__str__()
    def B(self,r,s,coord):
        B = np.zeros((2, self.ndofs))
        dhdx = self.dhdx(r,s)
        jaco_inv = self.inv_jacobian(dhdx, coord)
        dhdx = np.dot(jaco_inv, dhdx)
        B[0, :] = dhdx[0, :]
        B[1, :] = dhdx[1, :]
        return B

class Quad4():
    def __init__(self,dofs_per_node):
        self.dofs_per_node = dofs_per_node
        self.nnodes = 4
        self.ngpoints = 4
        self._gpoints =  np.array([[-0.57735027,  0.57735027],
                                  [ 0.57735027,  0.57735027],
                                  [-0.57735027, -0.57735027],
                                  [ 0.57735027, -0.57735027]])
        self._gweights = np.ones(4)
    def __str__():
        return "Quad4"
    @property
    def gpoints(self):
        return self._gpoints
    @property
    def gweights(self):
        return self._gweights

    def dhdx(self,r,s):
        return 0.25*np.array([
                [s - 1, -s + 1, s + 1, -s - 1],
                [r - 1, -r - 1, r + 1, -r + 1]])

class Tria3:
    pass

class Tet10:
    pass

class Hex20:
    pass

_element_types = {1:Quad4,2:Tria3,3:Tet10,4:Hex20}
