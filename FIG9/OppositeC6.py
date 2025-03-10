import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator



# ns-ns
s0 =  11.97
s1 = -0.8486
s2 =  0.003385

# np-np
p0 = -0.2886
p1 =  0.0571
p2 = -0.000268

# nd-nd
d0 =  26.03
d1 =  0.01454
d2 =  66
n0 =  35.14

ns = np.arange(60, 100)
nd = np.arange(60, 100)
NS, ND = np.meshgrid(ns, nd)

RATIO = (NS/ND)**11 * (s0 + s1*NS + s2*ND**2) / (d0 + d1*ND + d2/(ND-n0))


os.makedirs('Data', exist_ok=True)

np.save('Data/NS.npy', NS)
np.save('Data/ND.npy', ND)
np.save('Data/Ratio.npy', RATIO)
quit()


