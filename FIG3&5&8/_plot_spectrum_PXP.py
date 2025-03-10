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


corol_r = 'darkred'
clrs = [(0.8, 0.8, 0.8), 'red']
cm = LinearSegmentedColormap.from_list("Custom", clrs, N=200)


Steps = 1000
V_min = 0.001
V_max = 1
fib = [5, 8, 13]



Omega0 = 8   * 2*pi
Delta  = Omega0*3
# T      = 1
# dev    = 1/3 * T


fig, axs = plt.subplots(2, 3, figsize=(17.5/2.54, 7./2.54))
((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs
fig.subplots_adjust(left=0.12, right=0.95, top=0.928, bottom=0.13, hspace=0.12, wspace=0.12)



colormap_name = cm
alpha = 1

for ax in axs.reshape(6):
    ax.axis([-1.1 * Delta, 1.1 * Delta, -3.15 * Delta, 3.15 * Delta])
    ax.set_xticks([-Delta, 0, Delta])
    ax.set_yticks([-3. * Delta, -2. * Delta, -Delta, 0, Delta, 2. * Delta, 3. * Delta])

for ax in axs[1]:
    ax.set_xticklabels([r'$-\Delta_0$', '0', r'$\Delta_0$'])
for ax in axs[0]:
    ax.set_xticklabels([])

for ax in axs[:, 0]:
    ax.set_yticklabels([r'$-3\Delta_0$', '', '', '0', '', '', r'$3\Delta_0$'])
for ax in axs[:, 1]:
    ax.set_yticklabels([])
for ax in axs[:, 2]:
    ax.set_yticklabels([])

ax0 = fig.add_subplot(111, frameon=False)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlabel('Detuning $\Delta$', fontsize=10, labelpad=16, x=0.483)
ax0.set_ylabel('Eigenenergies $\mathcal{E}_k$', fontsize=10, labelpad=30)



fig.text(0, 0.8, ' Step I', fontsize=10)
fig.text(0, 0.22, 'Step II', fontsize=10)


axs[0, 0].set_title(r'$\nu=3$', fontsize=10)
axs[0, 1].set_title(r'$\nu=4$', fontsize=10)
axs[0, 2].set_title(r'$\nu=5$', fontsize=10)

axs[0, 0].text(0.65*Delta, -2.6*Delta, r'$|A_3\rangle$', fontsize=8)
axs[0, 0].text(-0.95*Delta, -1.*Delta, r'$|G_3\rangle$', fontsize=8)
axs[0, 1].text(0.65*Delta, -2.6*Delta, r'$|A_4\rangle$', fontsize=8)
axs[0, 1].text(-0.95*Delta, -1.*Delta, r'$|G_4\rangle$', fontsize=8)
axs[0, 2].text(0.25*Delta, -2.8*Delta, r'$-|A_5\rangle$', fontsize=8)
axs[0, 2].text(-0.95*Delta, -1.*Delta, r'$|G_5\rangle$', fontsize=8)


axs[1, 0].text(-0.95*Delta, 2.3*Delta, r'$|A_3\rangle$', fontsize=8)
axs[1, 0].text(0.7*Delta, 0.5*Delta, r'$|G_3\rangle$', fontsize=8)
axs[1, 1].text(-0.95*Delta, 2.3*Delta, r'$|A_4\rangle$', fontsize=8)
axs[1, 1].text(0.7*Delta, 0.5*Delta, r'$|G_4\rangle$', fontsize=8)
axs[1, 2].text(-0.7*Delta, 2.4*Delta, r'$-|A_5\rangle$', fontsize=8)
axs[1, 2].text(0.6*Delta, 0.5*Delta, r'$-|G_5\rangle$', fontsize=8)




################################## N=3 ##############################################
detuning = np.load('Data_PXP/N3/detuning.npy')
vals     = np.load('Data_PXP/N3/vals.npy')
amp1     = np.load('Data_PXP/N3/amp1.npy')
amp2     = np.load('Data_PXP/N3/amp2.npy')


ax1.plot(detuning, vals[:, 0], corol_r, linewidth=1.2)
ax1.plot(detuning, vals[:, 2],   '--k', alpha=alpha/1.8, linewidth=0.5)
ax1.plot(detuning[(2*Steps)//5:(Steps*3)//5], vals[(2*Steps)//5:(Steps*3)//5, 0] - Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [1, 3, 4]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp1[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])
segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap     =colormap_name,
                                 array    =cols,
                                 linewidth=0.6,
                                 alpha    =alpha,
                                 norm     =colors.LogNorm(vmin=V_min, vmax=V_max))
line = ax1.add_collection(lc)


ax4.plot(detuning, vals[:, 4], corol_r, linewidth=1.2)
ax4.plot(detuning, vals[:, 2],   '--k', alpha=alpha/1.8, linewidth=0.5)
ax4.plot(-detuning[(2*Steps)//5:(Steps*3)//5], -vals[(2*Steps)//5:(Steps*3)//5, 0] + Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [0, 1, 3]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp2[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])
segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=colormap_name,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=alpha,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax4.add_collection(lc)


################################################################################ N=4
detuning = np.load('Data_PXP/N4/detuning.npy')
vals     = np.load('Data_PXP/N4/vals.npy')
amp3     = np.load('Data_PXP/N4/amp3.npy')
amp4     = np.load('Data_PXP/N4/amp4.npy')


ax2.plot(detuning, vals[:, 0], corol_r, linewidth=1.2)
ax2.plot(detuning, vals[:, 1], '--k', alpha=alpha/1.8, linewidth=0.5)
ax2.plot(detuning, vals[:, 4], '--k', alpha=alpha/1.8, linewidth=0.5)
ax2.plot(detuning, vals[:, 6], '--k', alpha=alpha/1.8, linewidth=0.5)
ax2.plot(detuning[(2*Steps)//5:(Steps*3)//5], vals[(2*Steps)//5:(Steps*3)//5, 0] - Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [2, 3, 5, 7]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp3[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])
segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=colormap_name,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=alpha,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax2.add_collection(lc)


ax5.plot(detuning, vals[:, 7], corol_r, linewidth=1.2)
ax5.plot(detuning, vals[:, 1],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax5.plot(detuning, vals[:, 4],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax5.plot(detuning, vals[:, 6],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax5.plot(-detuning[(2*Steps)//5:(Steps*3)//5], -vals[(2*Steps)//5:(Steps*3)//5, 0] + Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [0, 2, 3, 5]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp4[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])

segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=colormap_name,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=alpha,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax5.add_collection(lc)




################################################################################ N=5
detuning = np.load('Data_PXP/N5/detuning.npy')
vals     = np.load('Data_PXP/N5/vals.npy')
amp5     = np.load('Data_PXP/N5/amp5.npy')
amp6     = np.load('Data_PXP/N5/amp6.npy')

ax3.plot(detuning, vals[:, 0], corol_r, linewidth=1.2)
ax3.plot(detuning, vals[:, 2],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax3.plot(detuning, vals[:, 4],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax3.plot(detuning, vals[:, 7],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax3.plot(detuning, vals[:, 10],      '--k', alpha=alpha/1.8, linewidth=0.5)
ax3.plot(detuning[(2*Steps)//5:(Steps*3)//5], vals[(2*Steps)//5:(Steps*3)//5, 0] - Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [1, 3, 5, 6, 8, 9, 11, 12]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp5[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])

segments = np.vstack(segments)
cols     = np.hstack(cols)
lc = LineCollection(segments, cmap=colormap_name,
                    array=cols,
                    linewidth=0.6,
                    alpha=alpha,
                    norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax3.add_collection(lc)


ax6.plot(detuning, vals[:, 12], corol_r, linewidth=1.2)
ax6.plot(detuning, vals[:, 2],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax6.plot(detuning, vals[:, 4],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax6.plot(detuning, vals[:, 7],       '--k', alpha=alpha/1.8, linewidth=0.5)
ax6.plot(detuning, vals[:, 10],      '--k', alpha=alpha/1.8, linewidth=0.5)
ax6.plot(-detuning[(2*Steps)//5:(Steps*3)//5], -vals[(2*Steps)//5:(Steps*3)//5, 0] + Delta/2, 'k', linewidth=1.2)

segments = []
cols     = []
for ii in [0, 1, 3, 5, 6, 8, 9, 11]:
    x = np.real(detuning)
    y = np.real(vals[:, ii])
    cols.append(amp6[:, ii])
    segments.append([np.column_stack([x[i:i+2], y[i:i+2]]) for i in range(len(x) - 1)])

segments = np.vstack(segments)
cols     = np.hstack(cols)
lc   = LineCollection(segments,  cmap=colormap_name,
                                 array=cols,
                                 linewidth=0.6,
                                 alpha=alpha,
                                 norm=colors.LogNorm(vmin=V_min, vmax=V_max))
ax6.add_collection(lc)





############################### colorbar #####################################
col_bar = plt.colorbar(line, ax         =(ax1, ax2, ax3, ax4, ax5, ax6),
                             orientation='vertical',
                             fraction   =0.008, pad=0.028, aspect=30)
col_bar.ax.set_title(r'$\eta_{lk}$', fontsize=10)


plt.show()
