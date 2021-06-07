import solidspy.assemutil as ass
import solidspy.solutil as sol
import solidspy.postprocesor as pos
import solidspy.preprocesor as pre
import matplotlib.pyplot as plt

folder = "./examples/mesh/"
nodes, mats, elements, loads = pre.readin(folder=folder)

# assembly operator
DME , IBC , neq = ass.DME(nodes, elements)

# System assembly
KG = ass.assembler(elements, mats, nodes, neq, DME)
RHSG = ass.loadasem(loads, IBC, neq)

# solution
UG = sol.static_sol(KG, RHSG)

# post processing
UC = pos.complete_disp(IBC, nodes, UG)
pos.fields_plot(elements, nodes, UC)

# output
print(UC)
plt.show()