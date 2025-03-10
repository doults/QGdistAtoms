import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from math import *
params = {'text.usetex'         : True,
          'font.size'           : 8,
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



detuning = np.load('Data_vdW/detuning.npy')
omega = np.load('Data_vdW/omega.npy')
times0 = np.load('Data_vdW/times0.npy')

times3 = np.load('Data_vdW/times3.npy')
history3 = np.load('Data_vdW/history3.npy')

times4 = np.load('Data_vdW/times4.npy')
history4 = np.load('Data_vdW/history4.npy')

times5 = np.load('Data_vdW/times5.npy')
history5 = np.load('Data_vdW/history5.npy')

times_list = [times0, times3, times4, times5]






times_ls = [times3, times4, times5]
phase_ls = [np.angle(history3[:, 0]), np.angle(history4[:, 0]), np.angle(history5[:, 0])]
ground_ls = [np.abs(history3[:, 0])**2, np.abs(history4[:, 0])**2, np.abs(history5[:, 0])**2]






fig, axs = plt.subplots(3, 1, figsize=(8/2.5, 8.1/2.5))
fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.09, hspace=0.34)



for ii, (times, ground, phases) in enumerate(zip(times_ls, ground_ls, phase_ls)):
    ax = axs[ii]
    ymax = 1.5
    ax.set_ylim(-0.08, 1.3)
    ax.set_xlim(-0.05, 2.05)

    ax.set_yticks([0., 0.5, 1.])
    ax.set_yticklabels(['0', '', '1'])


    ax.yaxis.set_minor_locator(FixedLocator([0.5 / 3, 1. / 3, 4 * 0.5 / 3, 5 * 0.5 / 3]))
    ax.text(x=3.4/2, y=ymax*0.93, s=rf'$\nu={ii+3}$')



    inset_ax = ax.inset_axes([0.114, .825, 0.297, 0.42])  # [left, bottom, width, height]
    inset_ax.plot(times[:-25], np.angle(np.exp(1.j*(phases[:-25]+0.001))), linewidth=0.5, alpha=0.9)
    for axis in ['top', 'bottom', 'left', 'right']:
        inset_ax.spines[axis].set_linewidth(0.5)
    inset_ax.tick_params(axis='both', which='both', labelsize=4.6, pad=1.2, width=0.5)
    inset_ax.tick_params(axis='both', which='major', length=2.7)
    inset_ax.tick_params(axis='both', which='minor', length=1.5)

    inset_ax.set_ylim(-pi-0.9, pi + 0.9)
    inset_ax.set_yticks([-pi, 0, pi])
    inset_ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    inset_ax.xaxis.set_minor_locator(MultipleLocator(1/3))
    inset_ax.set_ylabel('$\phi$', fontsize=6, labelpad=-2)
    inset_ax.set_xlabel('$t$', fontsize=6, labelpad=1)


    bbox = inset_ax.get_position()
    rectangle = patches.FancyBboxPatch((bbox.x0 - bbox.width*0.27, bbox.y0), bbox.width*1.33, bbox.height,
                                       boxstyle="square,pad=0", transform=fig.transFigure, edgecolor='none',
                                       facecolor='w', zorder=-1, clip_on=False)
    inset_ax.add_patch(rectangle)


for ax in axs[:-1]:
    ax.set_xticklabels([])






axs[-1].set_xlabel('Time $t$ [$\mu s$]', labelpad=1)
axs[1].set_ylabel(r'Populations')
axs[-1].tick_params(axis='both', which='both', pad=3.)


axs[0].plot(times3, np.abs(history3[:, 0])**2, color='k', alpha=0.8, lw=0.9)
axs[0].plot(times3, np.abs(history3[:, int('101', 2)])**2, 'tab:red', alpha=0.65, lw=0.9)


axs[1].plot(times4, np.abs(history4[:, 0])**2, color='k', alpha=0.8, linewidth=0.9)
axs[1].plot(times4, np.abs(history4[:, int('1001', 2)])**2, 'tab:red', alpha=0.65, linewidth=0.9)
axs[1].plot(times4, np.abs(history4[:, int('0101', 2)] + history4[:, int('1010', 2)])**2/2, '--',
            color='tab:red', alpha=0.65, linewidth=0.9)



axs[2].plot(times5, np.abs(history5[:, 0])**2, color='k', alpha=0.8, linewidth=0.9)
axs[2].plot(times5, np.abs(history5[:, int('10101', 2)])**2, color='tab:red', alpha=0.65, linewidth=0.9)


axs[0].text(x=3.35/2, y=0.94, s=r'$|G_3\rangle$', fontsize=8)
axs[1].text(x=3.35/2, y=0.94, s=r'$|G_4\rangle$', fontsize=8)
axs[2].text(x=3.35/2, y=0.94, s=r'$|G_5\rangle$', fontsize=8)

axs[0].text(x=2.65/2, y=0.72, s=r'$|A_3\rangle$', fontsize=8, color='darkred')
axs[1].text(x=1.85/2, y=0.12, s=r'$|\aleph_2\rangle$', fontsize=8, color='darkred')
axs[1].text(x=1.65/2, y=0.73, s=r'$|A_4\rangle = |\aleph_1\rangle$', fontsize=8, color='darkred')

axs[2].text(x=2.54/2, y=0.72, s=r'$|A_5\rangle$', fontsize=8, color='darkred')

plt.show()