import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

elem_state = np.load("out.npy")
ndiv=int(np.sqrt(len(elem_state)))
print(ndiv)
fig,axs = plt.subplots(1,1)
colors = ["white", "grey","grey","blue"]
nodes = [0.0, 0.4, 0.6,1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

axs.imshow(elem_state.reshape((ndiv,ndiv)),cmap=cmap,origin="lower")
plt.show()