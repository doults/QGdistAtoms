import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap


params = {'text.usetex'         : True,
          'font.size'           : 7,
          'font.family'         : 'modern',
          'xtick.top'           : True,
          'ytick.right'         : True,
          'ytick.direction'     : 'inout',
          'xtick.direction'     : 'inout',
          'xtick.minor.visible' : True,
          'ytick.minor.visible' : True,
          }
plt.rcParams.update(params)



############################# DATA ##############################
Ratios = np.load('Data/Ratio.npy')
Deltas = np.load('Data/delta_sd_B2.npy')
Edge = np.load('Data/Edge.npy')
Bulk = np.load('Data/Bulk.npy')
NS = np.load('Data/NS.npy')
ND = np.load('Data/ND.npy')



############################# FIG ###############################
fig = plt.figure(figsize=(8.5 / 2.54, 8.5/2.54))
gs = gridspec.GridSpec(2, 2, height_ratios=[0.9, 0.75], width_ratios=[0.3, 0.7])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

plt.subplots_adjust(left=0.15,
                    right=0.915,
                    top=0.96,
                    bottom=0.11, hspace=0.36, wspace=0.3)


############################## ax1 ##############################
ax1.axis('off')

############################## ax2 ##############################
clrs = [(1, 1, 1),
        (0.95, 0.95, 1),
        (0.9, 0.9, 1),
        (0.8, 0.8, 1),
        (0.65, 0.65, 1),
        (0.5, 0.5, 1),
        (0., 0., 0),
        (0.5, 0.5, 1),
        (0.65, 0.65, 1),
        (0.8, 0.8, 1),
        (0.9, 0.9, 1),
        (0.95, 0.95, 1),
        (1, 1, 1)]

cm = LinearSegmentedColormap.from_list("Custom", clrs, N=400)
c = ax2.pcolor(NS, ND, -Ratios, shading='auto',
               vmin   =0.8,
               vmax   =1.2,
               cmap   =cm)
cbar = fig.colorbar(c, ax=ax2, fraction=0.028, pad=0.08, aspect=21)
cbar.ax.set_title(r"$-C_6/C'_6$", fontsize=7, pad=10)
cbar.ax.title.set_position((3, 1))

ax2.set_xlim(57, 103)
ax2.set_ylim(57, 103)

ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.tick_params(which='both', direction='inout')
ax2.xaxis.set_minor_locator(MultipleLocator(5))
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.xaxis.set_major_locator(MultipleLocator(10))

ax2.set_ylabel(r'$n_S$', fontsize=8, labelpad=1)
ax2.set_xlabel(r'$n_D$', fontsize=8, labelpad=1)
ax2.grid(alpha=0.1, which='both')
ax2.grid(alpha=0.4)


############################## ax3 ##############################
Block = 40 * 2 * np.pi
B2 = Block / 64
Omega_ar = 8 * 2 * np.pi
deltas_theo = np.linspace(B2, -3.5 * B2, 1000)

edge_theo = 1 - (B2 + deltas_theo) ** 2 / np.abs(Omega_ar) ** 2
bulk_theo = 1 - (2 * B2 + deltas_theo) ** 2 / np.abs(Omega_ar) ** 2

ax3.plot(deltas_theo / B2, edge_theo, linestyle='--', color='darkblue', lw=0.9)
ax3.plot(deltas_theo / B2, bulk_theo, linestyle='--', color='darkred', lw=0.9)
ax3.scatter(Deltas[::2], Edge[::2], facecolors='white', edgecolors='darkblue', zorder=1, label='Edge', alpha=0.8, s=7)
ax3.scatter(Deltas[::2], Bulk[::2], marker='^', facecolors='white', edgecolors='darkred', zorder=1, label='Bulk', s=7,
            alpha=0.8)


ax3.set_ylim(0.979, 1.001)
ax3.set_xlim(-3.075, 0.075)

ax3.xaxis.set_minor_locator(MultipleLocator(0.5 / 2))
ax3.yaxis.set_minor_locator(MultipleLocator(0.01 / 4))
ax3.yaxis.set_major_locator(MultipleLocator(0.01))

ax3.set_xlabel(r'$\delta_{SD}/B_2$', fontsize=8)
ax3.set_ylabel(r'Transfer Probability', fontsize=8)
ax3.legend(fontsize=6)

pos2 = ax2.get_position()
pos3 = ax3.get_position()
right_x_position = pos2.x0 + pos2.width
ax3.set_position([pos3.x0, pos3.y0, right_x_position-pos3.x0, pos3.height])

plt.rcParams['text.usetex'] = False
fig.text(0.006, 0.97, '(a)', fontsize=8)
fig.text(0.32, 0.97, '(b)', fontsize=8)
fig.text(0.006, 0.46, '(c)', fontsize=8)

plt.show()
