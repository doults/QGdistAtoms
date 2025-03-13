from QMOptimize import *
from matplotlib.ticker import MultipleLocator
import os


params = {'text.usetex'         : True,
          'font.size'           : 6,
          'font.family'         : 'modern',
          'xtick.top'           : True,
          'ytick.right'         : True,
          'ytick.direction'     : 'inout',
          'xtick.direction'     : 'inout',
          'xtick.minor.visible' : True,
          'lines.markersize'    : 3.5
          }
plt.rcParams.update(params)


N       = 5
B       = 40  * 2*pi
Omega_0 = 8   * 2*pi
Delta_0 = 16  * 2*pi
Gamma   = 0.5 * 1e-3 * 2*pi



current_dir  = os.getcwd()
improved_dir = current_dir + '/Data_N5/improved/B40_Omega8_Delta16/'
disorder_dir = current_dir + '/Data_N5/disordered/B40_Omega8_Delta16/'

vert1 = 32.1596
vert2 = 34.0513
vert3 = 36.6998



fid_dis      = np.load(disorder_dir + 'variance0.01/fidelities.npy')
fid_dis_ave    = np.average(fid_dis, axis=1)[::-1]
fid_dis_errors = np.std(fid_dis, axis=1)[::-1]

areas, fids = [], []
times1, times2, times3 = 0, 0, 0
pulse0, pulse1, pulse2, pulse3 = 0, 0, 0, 0
for directory in [name for name in sorted(os.listdir(improved_dir))]:
    area = float(directory[4:])
    areas.append(area)
    fid     = np.load(improved_dir + directory + '/Fidelity.npy')
    opt_par = np.load(improved_dir + directory + '/Opt_par.npy')


    system = Hamiltonians(num_of_atoms=N)
    system.chain()
    system.rydberg_3D_array_vdw(block=B, decay_r=0, power=6, flip_b=True)
    T = area / Omega_0 / 0.7527
    system.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2/2.5, power=8)
    optimizer = Optimizer()
    optimizer.configure(input_states =[],
                        target_states=[],
                        initial_guess=opt_par,
                        factors      =[],
                        resolution   =500,
                        systems      =[system])
    optimizer.update_pulses()
    if area == vert1:
        times1 = system.times/system.times[-1]*2
        pulse1 = system.pulses[1](system.times)

    if area == vert2:
        times2 = system.times/system.times[-1]*2
        pulse2 = system.pulses[1](system.times)

    if area == vert3:
        pulse0 = system.pulses[0](system.times)
        times3 = system.times/system.times[-1]*2
        pulse3 = system.pulses[1](system.times)

    fids.append(fid)

fids = np.array(fids)
two_tau = np.array(areas) / Omega_0 / 0.7527





Proj = np.array([[0, 0], [0, 1]], dtype=np.complex128)
number_op1 = 0
number_op2 = 0
number_op3 = 0

system = Hamiltonians(num_of_atoms=N-2)
for ii in range(N-2):
    number_op1 += system.__expand__(matrix=Proj, from_subspace=ii)
system = Hamiltonians(num_of_atoms=N - 1)
for ii in range(N-1):
    number_op2 += system.__expand__(matrix=Proj, from_subspace=ii)
system = Hamiltonians(num_of_atoms=N)
for ii in range(N):
    number_op3 += system.__expand__(matrix=Proj, from_subspace=ii)




times = np.load(disorder_dir + '/variance0.0/times.npy')
decay_00 = []
for ii, phi_00 in enumerate(np.load(disorder_dir + 'variance0.0/histories_00.npy')):
    integrand = np.einsum('ti,ij,tj->t', phi_00, number_op1, phi_00.conj())
    decay_00.append(np.trapz(integrand, x=times[ii]))
decay_00 = np.array(decay_00)

decay_01 = []
for ii, phi_01 in enumerate(np.load(disorder_dir + 'variance0.0/histories_01.npy')):
    integrand = np.einsum('ti,ij,tj->t', phi_01, number_op2, phi_01.conj())
    decay_01.append(np.trapz(integrand, x=times[ii]))
decay_01 = np.array(decay_01)

decay_10 = []
for ii, phi_10 in enumerate(np.load(disorder_dir + 'variance0.0/histories_10.npy')):
    integrand = np.einsum('ti,ij,tj->t', phi_10, number_op2, phi_10.conj())
    decay_10.append(np.trapz(integrand, x=times[ii]))
decay_10 = np.array(decay_10)

decay_11 = []
for ii, phi_11 in enumerate(np.load(disorder_dir + 'variance0.0/histories_11.npy')):
    integrand = np.einsum('ti,ij,tj->t', phi_11, number_op3, phi_11.conj())
    decay_11.append(np.trapz(integrand, x=times[ii]))
decay_11 = np.array(decay_11)


decay_errors = np.exp(-Gamma * 1/4 * (decay_00 + decay_01 + decay_10 + decay_11))



################################################################################################
fig = plt.figure(figsize=(8.5/2.5, 7.5/2.5))
gs = GridSpec(2, 3, height_ratios=[2, 1])
plt.subplots_adjust(left=0.128,
                    right=0.918,
                    top=0.9,
                    bottom=0.09, hspace=0.45, wspace=0.15)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

ax1_up = ax1.twiny()
ax2_right = ax2.twinx()
ax3_right = ax3.twinx()
ax4_right = ax4.twinx()


inset = inset_axes(ax1, width="100%", height="100%",
                   bbox_to_anchor=(0.625, 0.61, 0.3625, 0.38),
                   bbox_transform=ax1.transAxes)
inset.set_xlabel(r'$\tau$', fontsize=5.5, labelpad=1)
inset.tick_params(axis='both', which='major', labelsize=4.5, width=0.5, length=2, pad=2)
inset.tick_params(axis='both', which='minor', width=0.4, length=1.3)
for spine in inset.spines.values():
    spine.set_linewidth(0.5)
inset.set_yscale('log')
inset.set_ylabel(r'$1-F$', fontsize=5, labelpad=1)
inset.set_xticks([0.40, 0.45, 0.5])
inset.xaxis.set_minor_locator(MultipleLocator(0.5/3))


ax1.plot(two_tau/2, 1-np.array(fids), color='darkblue', lw=0.6, zorder=1)
ax1.scatter(two_tau/2, 1-np.array(fids), s=5, lw=0.6, facecolors='white', edgecolors='darkblue', alpha=1, zorder=2)
ax1.plot(two_tau/2, 1-fid_dis_ave, color='darkred', lw=0.6, zorder=1)
ax1.scatter(two_tau/2, 1-fid_dis_ave, s=5, lw=0.6, marker='^', facecolors='white', edgecolors='darkred', alpha=1, zorder=2)
ax1.fill_between(two_tau/2, 1-fid_dis_ave, 1-fid_dis_ave+fid_dis_errors, color='darkred', alpha=0.2)

ax1_up.scatter(np.array(areas)/2, np.zeros_like(areas))
ax1_up.vlines([vert1/2, vert2/2], ymin=1e-9, ymax=100, lw=0.3, linestyles='--', color='k')
ax1_up.vlines([vert3/2], ymax=4e-4, ymin=1e-9, lw=0.3, linestyles='--', color='k')

ax1.set_yscale('log', base=10)
ax1.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,  1e-1, 1])
ax1.set_yticklabels(['$10^{-8}$', '', '$10^{-6}$', '', '$10^{-4}$', '', '$10^{-2}$', '', '$10^0$'])
ax1.set_ylim(2e-8, 30)
ax1.set_ylabel('Gate Error', fontsize=7, labelpad=2)
ax1_up.set_xlabel(r'Pulse area $\theta$', fontsize=7)
ax1.set_xlabel(r'Pulse Duration $\tau[\mu \mathrm{s}]$', fontsize=7, labelpad=3)

inset.scatter(two_tau/2, 1-fid_dis_ave + 1 - decay_errors, s=0.4, lw=0.6, marker='^', color='darkred', alpha=1, zorder=3)
inset.scatter(two_tau/2, 1-np.array(fids) + 1 - decay_errors, s=0.4, lw=0.6, color='darkblue', alpha=1, zorder=3)
inset.set_ylim(6e-4, 2e-1)
inset.set_ylim(1e-3, 2e-2)





ax2_right.plot(times1[:500], pulse0[:500]/2/pi, lw=0.9, color='darkcyan', alpha=0.7)
ax3_right.plot(times2[:500], pulse0[:500]/2/pi, lw=0.9, color='darkcyan', alpha=0.7)
ax4_right.plot(times3[:500], pulse0[:500]/2/pi, lw=0.9, color='darkcyan', alpha=0.7)

ax2.plot(times1[:500], pulse1[:500]/2/pi, lw=0.8, color='darkred')
ax3.plot(times2[:500], pulse2[:500]/2/pi, lw=0.8, color='darkred')
ax4.plot(times3[:500], pulse3[:500]/2/pi, lw=0.8, color='darkred')

for ax in [ax2, ax3, ax4]:
    ax.spines['right'].set_color('darkcyan')
    ax.xaxis.set_minor_locator(MultipleLocator(0.5/3))
    ax.yaxis.set_minor_locator(MultipleLocator(15))
    ax.spines['left'].set_alpha(0)
    ax.tick_params(axis='y', colors='darkred', which='both')
    ax.set_ylim(-36, 36)
    ax.set_yticks([-30, 0, 30])
for ax in [ax2_right, ax3_right, ax4_right]:
    ax.spines['left'].set_color('darkred')
    ax.spines['right'].set_alpha(0)
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='y', colors='darkcyan', which='both')
    ax.set_ylim(-2/3, 8+2/3)
    ax.set_yticks([0, 4, 8])

ax3.set_xlabel(r'Time $t/\tau$', fontsize=7, labelpad=0)
ax2.set_ylabel('$\Delta /2\pi [\mathrm{MHz}]$', fontsize=7, color='darkred')
ax4_right.set_ylabel('$\Omega / 2\pi[\mathrm{MHz}]$', fontsize=7, color='darkcyan', labelpad=3)

ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax2_right.set_yticklabels([])
ax3_right.set_yticklabels([])

plt.rcParams['text.usetex'] = False
fig.text(0.006, 0.959, '(a)', fontsize=8)
fig.text(0.006, 0.34, '(b)', fontsize=8)


plt.show()
