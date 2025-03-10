from QMClasses import*
from math import*
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors


params = {'text.usetex'         : True,
          'font.size'           : 9,
          'font.family'         : 'modern',
          'xtick.top'           : True,
          'ytick.right'         : True,
          'ytick.direction'     : 'inout',
          'xtick.direction'     : 'inout',
          'xtick.minor.visible' : True,
          'ytick.minor.visible' : True,
          'markers.fillstyle'   : 'none',
          'lines.markersize'    : 3.5
          }
plt.rcParams.update(params)

N      = 4
Delta  = 18 * 2*pi
T      = 1
V_min = 10**(-3)
V_max = 10**(-0)
Steps = 1000

detunings = np.load('Data_vdW/N4/detuning.npy')
vals1 = np.load('Data_vdW/N4/vals1.npy')
amp1  = np.load('Data_vdW/N4/amp1.npy')
vals2 = np.load('Data_vdW/N4/vals2.npy')
amp2  = np.load('Data_vdW/N4/amp2.npy')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.3/2.5, 7.4/2.5))
fig.subplots_adjust(left=0.169, right=0.98, top=0.98, bottom=0.106, hspace=0.13, wspace=0.07)

clrs = [(0.7, 0.7, 0.7), 'red']
cm = LinearSegmentedColormap.from_list("Custom", clrs, N=200)


for ax in (ax1, ax2):
    ax.set_xlim(-1.07*Delta, 1.07*Delta)
    ax.set_ylim(-2.75*Delta, 2.75*Delta)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both', direction='inout')

    ax.set_xticks([-Delta, -1/2 * Delta, 0, Delta/2, Delta])
    ax.set_xticklabels([r'$-\Delta_0$', '', '0', '', r'$\Delta_0$'])

    ax.set_yticks([-2*Delta, -Delta, 0, Delta, 2*Delta])
    ax.set_yticklabels([r'$-2\Delta_0$', '', '0', '', r'$2\Delta_0$'])


ax1.text(x=-0.99 * Delta, y=-0.6 * Delta, s=r'$|G_4\rangle$', fontsize=8)
ax1.text(x=+0.4 * Delta, y=-2.25 * Delta, s=r'$|A_4\rangle = |\aleph_1\rangle$', fontsize=8)
ax1.text(x=+0.85 * Delta, y=-1.42 * Delta, s=r'$|\aleph_{2,3}\rangle$', fontsize=6)
ax1.text(x=-0.99 * Delta, y=1.44 * Delta, s=r'$|\aleph_1\rangle$', fontsize=6)
ax1.text(x=-0.79 * Delta, y=1.96 * Delta, s=r'$|\aleph_{2,3}\rangle$', fontsize=6)

ax2.text(x=+0.85 * Delta, y=+0.31 * Delta, s=r'$|G_4\rangle$', fontsize=8)
ax2.text(x=+0.54 * Delta, y=-2.11 * Delta, s=r'$|\aleph_{2,3}\rangle$', fontsize=6)
ax2.text(x=+0.85 * Delta, y=-1.48 * Delta, s=r'$|\aleph_1\rangle$', fontsize=6)

ax2.text(x=-1.02 * Delta, y=1.25 * Delta, s=r'$|\aleph_{2,3}\rangle$', fontsize=6)
ax2.text(x=-0.79 * Delta, y=1.96 * Delta, s=r'$|A_4\rangle = |\aleph_1\rangle$', fontsize=8)

ax1.text(x=Delta / 1.5, y=Delta * np.ceil(N / 2) - 0.3 * Delta, s='$B>0$')
ax2.text(x=Delta / 1.5, y=Delta * np.ceil(N / 2) - 0.3 * Delta, s='$B<0$')
ax1.set_xticklabels([])

ax0 = fig.add_subplot(111, frameon=False)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel(r'Eigenenrgies $\mathcal{E}_k$', labelpad=29, x=0.483, fontsize=10)
ax0.set_xlabel(r'Detuning $\Delta$', labelpad=13, fontsize=10)


################################## ax1 ########################################
ax1.plot(detunings, vals1[:, 1], '--k', alpha=1 / 1.8, linewidth=0.5)
ax1.plot(detunings, vals1[:, 3], '--k', alpha=1 / 1.8, linewidth=0.5)
ax1.plot(detunings, vals1[:, 6], '--k', alpha=1 / 1.8, linewidth=0.5)

segments = []
cols     = []
for ii in [2, 4, 5, 7]:
    x = np.real(detunings)
    y = np.real(vals1[:, ii])
    cols.append(amp1[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])

segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=cm,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=1,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax1.add_collection(lc)
ax1.plot(detunings, vals1[:, 0], 'darkred', linewidth=1.2)
ax1.plot(detunings[(2 * Steps) // 5:(Steps * 3) // 5], vals1[(2 * Steps) // 5:(Steps * 3) // 5, 0] - Delta / 2, 'k', linewidth=1.2)



################################## ax2 ########################################
ax2.plot(-detunings[(2 * Steps) // 5:(Steps * 3) // 5], -vals1[(2 * Steps) // 5:(Steps * 3) // 5, 0] + Delta / 2, 'k', linewidth=1.2)
ax2.plot(detunings, vals2[:, -2], '--k', alpha=1 / 1.8, linewidth=0.5)
ax2.plot(detunings, vals2[:, -5], '--k', alpha=1 / 1.8, linewidth=0.5)
ax2.plot(detunings, vals2[:, -7], '--k', alpha=1 / 1.8, linewidth=0.5)

segments = []
cols     = []
for ii in [-3, -4, -6, -8]:
    x = np.real(detunings)
    y = np.real(vals2[:, ii])
    cols.append(amp2[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])

segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=cm,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=1,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax2.add_collection(lc)
ax2.plot(detunings, vals2[:, -1], 'darkred', linewidth=1.2)


plt.show()