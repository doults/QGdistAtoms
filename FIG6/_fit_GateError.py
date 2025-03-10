from functools import partial
from QMClasses import *
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

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


def LZ_error(dur, _c_nu, _mu):
    return np.log(_mu) - _c_nu*dur


#########################################################
# Gamma  = np.round(np.array([0.0005])*2*3.1415, 4)
# Block = round(45 * 2*3.1415, 4)
# Omega = round(8  * 2*3.1415, 4)
# Delta = round(20 * 2*3.1415, 4)
#########################################################


Gamma = 0.0005 * 2*pi
Block = 45 * 2*pi
Omega = 8  * 2*pi
Delta = 20 * 2*pi


print('n', '\t c_n', '\t\t\t\t T_opt', '\t\t E_min')
print('-------------------------------------------------')
Cnu = []
Topts = []
Emins = []
for N in range(3, 9):
    path = f'Data_tot/N{N}/decay{0:.5f}_block{Block:.2f}_omega{Omega:.3f}_delta{Delta:.3f}'
    dimless_durations = np.load(path + f'/durations.npy')/2 * Omega**2/Delta
    errors    = np.load(path + f'/errors.npy')

    nu_r = N/2 - 0.25
    mu   = 2 - N%2

    p    = curve_fit(f=partial(LZ_error, _mu=mu), xdata=dimless_durations[:-9], ydata=np.log(errors[:-9]))
    c_nu = p[0][0]
    Topt = 1 / c_nu / (Omega**2 / Delta) * np.log(c_nu * mu/nu_r * (Omega**2 / Delta) / Gamma)
    Emin = nu_r/c_nu * Gamma / (Omega**2 / Delta) * (1 + np.log(c_nu * mu/nu_r * (Omega**2 / Delta) / Gamma))

    print(N, '\t', np.round(p[0][0], 3), '±', np.sqrt(p[1][0][0]).round(3), '\t\t',
          Topt.round(2), '\t\t', Emin.round(5))
    Cnu.append(p[0][0])
    Topts.append(Topt)
    Emins.append(Emin)
print('-------------------------------------------------')


# #########################################################
fig, ax  = plt.subplots(1, 1, figsize=(8/2.5, 5.6/2.5))
fig.subplots_adjust(left=0.163, right=0.98, top=0.98, bottom=0.13)
inset_ax = ax.inset_axes([0.44, .58, 0.53, 0.39])  # [left, bottom, width, height]



Gamma = 0.0005 * 2*pi
Block = 45 * 2*pi
Omega = 8  * 2*pi
Delta = 20 * 2*pi
t = np.linspace(0, 3, 1000)
colors = ['darkred', 'darkgreen', 'darkblue', 'darkorange', 'indigo', 'darkcyan']
for N, col, c_nu, Topt, Emin in  zip([3, 4, 5, 6, 7, 8], colors, Cnu, Topts, Emins):
    path = f'Data_tot/N{N}/decay{Gamma:.5f}_block{Block:.2f}_omega{Omega:.3f}_delta{Delta:.3f}'
    durations = np.load(path + f'/durations.npy')/2
    errors    = np.load(path + f'/errors.npy')
    ax.scatter(durations, errors, marker='o', alpha=0.9, color=col, lw=0.7, s=6)

    nu_r = N / 2 - 1 / 4
    mu   = 2 - N%2
    f = nu_r * Gamma * t + np.exp(LZ_error(t, _c_nu=c_nu * Omega**2/Delta, _mu=mu))

    ax.plot(t, f, c=col, linewidth=0.6, alpha=0.65)
    ax.plot(Topt, Emin, color='r', marker='x', ms=4, mew=1.)

    lambda1 = Omega/Block
    lambda2 = Delta/Block

    path = f'Data_opt/N{N}/ratio1{lambda1:.4}_ratio2{lambda2:.4}'
    errors_opt = np.load(path + '/errors.npy')
    B_div_gam = np.load(path + '/B_div_Gamma.npy')
    inset_ax.scatter(B_div_gam, errors_opt, c=col, marker='x', alpha=0.65, s=4, linewidth=0.8)

    xx = np.logspace(4, 8, 100)
    yy = nu_r * lambda2 / lambda1**2 / c_nu / xx * (1 + np.log(c_nu * mu / nu_r * lambda1**2 / lambda2 * xx))
    inset_ax.plot(xx, yy, c=col, linewidth=0.4, alpha=0.65)



ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.5/4))

inset_ax.set_ylabel('$E_\mathrm{min}$', fontsize=6, labelpad=2)
inset_ax.set_xlabel('$|B|/\Gamma$', fontsize=6,  labelpad=2)
for axis in ['top', 'bottom', 'left', 'right']:
    inset_ax.spines[axis].set_linewidth(0.5)
inset_ax.tick_params(axis='both', which='both', labelsize=4.5, pad=1.7, width=0.5)
inset_ax.tick_params(axis='both', which='major', length=2.7)
inset_ax.tick_params(axis='both', which='minor', length=1.5)
inset_ax.set_yscale('log')
inset_ax.set_xscale('log')

inset_ax.set_xlim(1.5*10**4, 1.5*10**7)
inset_ax.set_ylim(0.00015, 0.15)

ax.set_yscale('log')
ax.set_ylim(0.0015, 1.5)
ax.set_xlim(0.45, 2.05)

ax.set_ylabel('Error Probability $(1 -F)$', fontsize=8)
ax.set_xlabel(r'$\tau[\mu s]$ ', fontsize=8, labelpad=-1)

ax.text(x=0.60, y=0.0030, s='$N=3$', fontsize=6.5)
ax.text(x=0.75, y=0.0077, s='$4$',   fontsize=6.5)
ax.text(x=0.82, y=0.0107, s='$5$',   fontsize=6.5)
ax.text(x=0.83, y=0.0280, s='$6$',   fontsize=6.5)
ax.text(x=0.95, y=0.0250, s='$7$',   fontsize=6.5)
ax.text(x=0.60, y=0.2000, s='$8$',   fontsize=6.5)

plt.show()