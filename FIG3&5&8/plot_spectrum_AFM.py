from QMClasses import *
from math import *
from matplotlib.ticker import MultipleLocator

params = {'text.usetex'         : True,
          'font.size'           : 5,
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


Omega0 = 2  * 2*pi
Delta  = 20 * 2*pi
B      = 2.  * Delta

fig, (axs1, axs2) = plt.subplots(2, 3, figsize=(8.5/2.54, 5.2/2.54))
fig.subplots_adjust(left=0.135, right=0.985, top=0.91, bottom=0.16, hspace=0.2, wspace=0.13)


for ii, nn in enumerate([4, 6, 8]):
    system = Hamiltonians(num_of_atoms=nn)
    system.constDelta_linOmega(delta=Delta, amplitude=Omega0, duration=1, steps=100)
    system.chain()

    simulator = QuantumSimulator()
    simulator.times = system.times

    system.pxp_sw_nnn(b=1000)
    vals_pxp, vecs_pxp = simulator.spectrum(system.hamiltonian)
    system.rydberg_3D_array_vdw(block=B, power=6)
    vals_vdw, vecs_vdw,  = simulator.spectrum(system.hamiltonian)

    omega = system.pulses[0](system.times)
    delta = system.pulses[1](system.times)


    for kk, color, lw in zip(range(1, nn//2 + 2), ['darkred', 'grey', 'red', 'grey', 'red'], [1.4, .9, .9, .9, .9]):
        Sigma = np.abs(omega)**2/4/delta
        Sigma_tilde = np.abs(omega)**2/4/(delta-B/64)

        nur = np.ceil(nn/2)
        Ek0 = -nur * Delta
        Ek0_v = Ek0 + (nur-2)*B/64 + B/3**6
        dEk = -nur * Sigma - 2*Sigma * np.cos(kk*pi/(nn/2+2))

        if color == 'grey': dashes = (7, 1)
        else:               dashes = (5, 0)

        ratio = omega/delta
        pxp_theory = (np.real(vals_pxp[:, kk - 1]) / Delta - Ek0 / Delta)
        vdw_theory = (np.real(vals_vdw[:, kk - 1]) / Delta - Ek0_v / Delta)

        axs1[ii].plot(ratio, pxp_theory * 100, dashes=dashes, alpha=0.8, linewidth=lw, color=color)
        axs2[ii].plot(ratio, vdw_theory * 100, dashes=dashes, alpha=0.8, linewidth=lw, color=color)
        axs1[ii].plot(ratio, dEk/Delta*100 , ':k', linewidth=0.9, dashes=(1, 2))

        shift_1 = -np.abs(omega)**2 / 4 / delta
        shift_2 = -np.abs(omega)**2 / 4 / (B - delta)
        shift_3 = -np.abs(omega)**2 / 4 / (2 * B - delta)
        sigma_1 = shift_1
        sigma_2 = shift_2
        if 0 < kk < int(nn / 2):
            dEk_v = nur * shift_1 + 2*shift_2 + (nur - 2)*shift_3 + 2 * (sigma_1 + sigma_2) * np.cos(2 * kk * pi / nn)
            axs2[ii].plot(ratio, dEk_v / Delta*100, ':k', linewidth=0.9, dashes=(1, 2))
        else:
            dEk_v = nur * shift_1 + 1 * shift_2 + (nur - 1) * shift_3 + B/64-B/3**6
            axs2[ii].plot(ratio, dEk_v / Delta * 100, ':k', linewidth=0.9, dashes=(1, 2))


    axs1[ii].set_title(rf'$\nu={nn}$', fontsize=8)
    axs1[ii].set_ylim(-0.85, 0.05)
    axs2[ii].set_ylim(-2.25, 3.25)
    axs2[ii].set_yticks([-2., -1., 0., 1., 2., 3.])


    axs1[ii].set_xticklabels([])
    axs1[ii].yaxis.set_minor_locator(MultipleLocator(0.2/2))
    axs2[ii].yaxis.set_minor_locator(MultipleLocator(1/2))

    axs2[ii].set_xticks([0, 0.05, 0.1])
    axs2[ii].set_xticklabels(['0.0', '0.05', '0.1'])


ax0 = fig.add_subplot(111, frameon=False)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlabel(r'$|\Omega|/\Delta$', fontsize=8, labelpad=13, x=0.483)
ax0.set_ylabel(r'$\delta\mathcal{E}_k/\Delta\times 10^2$', fontsize=8, labelpad=20)

axs1[1].set_yticklabels([])
axs1[2].set_yticklabels([])
axs2[1].set_yticklabels([])
axs2[2].set_yticklabels([])

plt.show()