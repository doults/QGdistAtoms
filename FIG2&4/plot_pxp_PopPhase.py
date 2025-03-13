import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import matplotlib.patches as patches


import numpy as np
from math import*
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



detuning = np.load('Data_PXP/detuning.npy')
omega    = np.load('Data_PXP/omega.npy')
times0   = np.load('Data_PXP/times0.npy')

times3   = np.load('Data_PXP/times3.npy')
history3 = np.load('Data_PXP/history3.npy')

times4   = np.load('Data_PXP/times4.npy')
history4 = np.load('Data_PXP/history4.npy')

times5   = np.load('Data_PXP/times5.npy')
history5 = np.load('Data_PXP/history5.npy')



times_ls = [times3, times4, times5]
phase_ls = [np.angle(history3[:, 0]), np.angle(history4[:, 0]), np.angle(history5[:, 0])]
ground_ls     = [np.abs(history3[:, 0])**2, np.abs(history4[:, 0])**2, np.abs(history5[:, 0])**2]


A3 = np.abs(history3[:, int('101', 2)])**2
A4 = np.abs((history4[:, int('0101', 2)] + history4[:, int('1010', 2)])/2 + history4[:, int('1001', 2)]/sqrt(2))**2
A5 = np.abs(history5[:, int('10101', 2)])**2
afm_ls = [A3, A4, A5]




fig, axs = plt.subplots(4, 1, figsize=(8/2.5, 9.8/2.5), gridspec_kw={'height_ratios': [1.1, 1, 1, 1]})
fig.subplots_adjust(left=0.12, right=0.92, top=0.96, bottom=0.076, hspace=0.34)



for ii, (times, ground, afm, phases) in enumerate(zip(times_ls, ground_ls, afm_ls , phase_ls)):
    ax = axs[ii+1]
    ymax = 1.5
    ax.set_ylim(-0.08, 1.3)
    ax.set_xlim([-0.05, 2.05])

    ax.set_yticks([0., 0.5, 1.])
    ax.set_yticklabels(['0', '', '1'])


    ax.yaxis.set_minor_locator(FixedLocator([0.5 / 3, 1. / 3, 4 * 0.5 / 3, 5 * 0.5 / 3]))


    ax.text(x=3.4/2, y=ymax*0.93, s=rf'$\nu={ii+3}$')
    ax.plot(times, ground, 'k', alpha=0.8, lw=0.9)
    ax.plot(times, afm, 'tab:red', alpha=0.65, lw=0.9)

    inset_ax = ax.inset_axes([0.114, .825, 0.297, 0.42])  # [left, bottom, width, height]
    inset_ax.plot(times[:-25], np.angle(np.exp(1.j*(phases[:-25]+0.001))), linewidth=0.5, alpha=0.9)
    for axis in ['top', 'bottom', 'left', 'right']:
        inset_ax.spines[axis].set_linewidth(0.5)
    inset_ax.tick_params(axis='both', which='both', labelsize=4.6, pad=1.2, width=0.5)
    inset_ax.tick_params(axis='both', which='major', length=2.7)
    inset_ax.tick_params(axis='both', which='minor', length=1.5)




    inset_ax.set_ylim(-pi-0.9, pi + 0.9)
    inset_ax.set_xlim(-0.16, 4.16/2)

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








twin2 = axs[0].twinx()
axs[0].plot(times0[:500], detuning[:500]/2/pi, 'darkblue', alpha=0.8)
axs[0].plot(times0[500:], detuning[500:]/2/pi, 'darkblue', alpha=0.8)
twin2.plot(times0, omega/2/pi, 'darkcyan', alpha=0.7, label='$\Omega(t)/\Omega_0$')



axs[0].set_yticks([-20, 0, 20],)
axs[0].set_ylim([-24, 24])
axs[0].set_xlim([-0.05, 2.05])
axs[0].yaxis.set_minor_locator(MultipleLocator(10))
axs[0].tick_params(axis='y', colors='darkblue')




twin2.set_yticks([0, 4, 8])
twin2.set_yticklabels(['0', '4', '8'])

twin2.yaxis.set_minor_locator(MultipleLocator(2))
twin2.set_ylim([-0.8, 8.8])

twin2.spines['right'].set_color('darkcyan')
axs[0].spines['right'].set_color('darkcyan')
twin2.tick_params(axis='y', colors='darkcyan')

twin2.spines['left'].set_color('darkblue')
axs[0].spines['left'].set_color('darkblue')


axs[0].set_ylabel('$\Delta/2\pi$ [MHz]', color='darkblue', labelpad=-1.5)
twin2.set_ylabel('$\Omega/2\pi$ [MHz]', color='darkcyan', labelpad=1.5)

axs[-1].set_xlabel('Time $t$ [$\mu s$]', labelpad=1)
axs[-1].tick_params(axis='both', which='both', pad=3.)


fig.text(x=0.004/2, y=0.973, s=r'\textbf{(a)}')
fig.text(x=0.004/2, y=0.73, s=r'\textbf{(b)}')


axs[1].text(x=3.3/2, y=0.94, s=r'$|G_3\rangle$', fontsize=8)
axs[2].text(x=3.3/2, y=0.94, s=r'$|G_4\rangle$', fontsize=8)
axs[3].text(x=3.3/2, y=0.94, s=r'$|G_5\rangle$', fontsize=8)

axs[1].text(x=2.70/2, y=0.72, s=r'$|A_3\rangle$', fontsize=8, color='darkred')
axs[2].text(x=2.62/2, y=0.72, s=r'$|A_4\rangle$', fontsize=8, color='darkred')
axs[3].text(x=2.60/2, y=0.72, s=r'$|A_5\rangle$', fontsize=8, color='darkred')

axs[2].set_ylabel('Populations')

plt.show()