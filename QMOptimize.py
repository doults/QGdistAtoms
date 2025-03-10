import time as tm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from QMClasses import*
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize


class Optimizer(QuantumSimulator):
    def __init__(self):
        super(Optimizer, self).__init__()
        self.status     = 'not configured'
        self.iterations = 0
        self.resolution  = None
        self.freq_cutoff = None
        self.rk_precision = 1e-9
        #################################
        self.systems       = None
        self.input_states  = None
        self.target_states = None
        self.phis          = None
        self.chis          = None
        self.factors       = None
        #################################
        self.operators  = None
        self.gradient   = None
        self.fidelity   = None
        self.fidelity_kind = None
        #################################
        self.opt_pulse      = None
        self.initial_pulse  = None
        self.Jacobians      = None
        self.opt_parameters = None



    def configure(self, systems, input_states, target_states, factors,
                  initial_guess=None, fidelity_kind='bell', mute_mask=(0, 1), resolution=1200,
                  freq_cutoff=None, rk_pre =1e-9):
        self.systems        = systems
        self.input_states   = input_states
        self.target_states  = target_states
        self.factors        = factors
        self.fidelity_kind  = fidelity_kind
        self.rk_precision   = rk_pre
        self.freq_cutoff    = freq_cutoff

        if initial_guess is None:
            self.opt_parameters = np.zeros((len(mute_mask), resolution), dtype=np.complex128)
        else:
            self.opt_parameters = initial_guess
        self.resolution     = resolution
        self.initial_pulse  = systems[0].pulses
        self.times          = np.linspace(systems[0].times[0], systems[0].times[-1], resolution)

        self.operators = []
        for _system in self.systems:
            _pulses_hold = _system.pulses[:]
            _operators_list = []
            for eye in np.eye(len(_system.pulses)):
                _system.pulses = [lambda time, mm=mm: (0.*time + mm) for mm in eye]
                _h1 = _system.hamiltonian(_system.times[0])
                _system.pulses = [lambda time: 0.*time               for _  in eye]
                _h2 = _system.hamiltonian(_system.times[0])
                _operators_list.append(_h1 - _h2)
            _system.pulses = _pulses_hold[:]
            self.operators.append(np.array(_operators_list))

        if mute_mask is not None:
            for qq in range(len(self.input_states)):
                self.operators[qq] = np.einsum('i, i...->i...', np.array(mute_mask), self.operators[qq])
        self.status = 'configured'

    def update_pulses(self):
        for ii in range(len(self.opt_parameters)):
            _opt_pulse = interp1d(self.times, self.opt_parameters[ii], kind='cubic', fill_value="extrapolate")
            for _system in self.systems:
                def new_pulse(time, idx=ii, _opt_pulse=_opt_pulse):
                    return self.initial_pulse[idx](time) + _opt_pulse(time)
                _system.pulses[ii] = new_pulse


    def generate_phis(self):
        """Returns |φ(t)> as a list of grid-data found via first using RK45 for
        the Shrödinger, then interpolate cubics.
        """
        self.phis = []
        for _input_state, _system in zip(self.input_states, self.systems):
            self.f_propagate(input_state=_input_state,
                             hamiltonian=_system.hamiltonian,
                             rtol=self.rk_precision,
                             atol=self.rk_precision)
            _grid = np.linspace(self.times[0].item(), self.times[-1].item(), self.resolution)
            _history_qubic = griddata(points=self.times,
                                      values=self.history,
                                      xi    =_grid,
                                      method='cubic')
            self.phis.append(_history_qubic)
        self.times = np.linspace(self.times[0].item(), self.times[-1].item(), self.resolution)
        return self.phis


    def generate_chis(self):
        """Returns |χ(t)> as a list of grid-data found via first using RK45
        for the Shrödinger with 'inverted' hamiltonian with the guessed pulses
                              Hⁱⁿᵛ(t) = H₀(T-t),
        then interpolate cubics.
        """
        self.chis = []
        for _target_state, _system in zip(self.target_states, self.systems):
            self.b_propagate(target_state=_target_state,
                             hamiltonian=_system.hamiltonian,
                             rtol=self.rk_precision,
                             atol=self.rk_precision)
            _grid = np.linspace(self.times[0].item(), self.times[-1].item(), self.resolution)
            _history_qubic = griddata(points=self.times,
                                      values=self.history,
                                      xi    =_grid,
                                      method='cubic')
            self.chis.append(_history_qubic)
        self.times = np.linspace(self.times[0].item(), self.times[-1].item(), self.resolution)
        return self.chis


    def fidelity_calculator(self, parameters=None, kind='bell', symmetric=True):
        """Calculates the fidelity F = |<ξ|U(T)|φ(0)>|² of preparing the system from |φ(0)> to
        |ξ>, as it evolves with U(t) = T exp{-i∫H₀(τ)dτ} and the gradient δF/δεᵢ(t) if the
        Hamiltonian H₀ is perturbed as
                H(t) = H₀(t) + ∑δεᵢ(t)Vᵢ.

        In this case the gradient is given by
                δF/δεᵢ(t) = 2δt Im{<χ(t)|Vᵢ|φ(t)><φ(t)|χ(t)>}

        Vᵢ    :     self.operators
        |φ(t)>:     self.phis,
        |χ(t)>:     self.chis
        """

        if parameters is not None:
            self.opt_parameters = parameters

        _chi_V_phis = []
        _phi_chis   = []

        for qq in range(len(self.input_states)):
            _input_operators = self.operators[qq]
            _chi_V_phi = np.einsum('ta,iab,tb->it', self.chis[qq].conj(), _input_operators, self.phis[qq])
            _chi_V_phi *= self.factors[qq]
            _phi_chi   = np.einsum('ta,ta->t', self.phis[qq].conj(), self.chis[qq])
            _phi_chi   *= self.factors[qq]
            _chi_V_phis.append(_chi_V_phi)
            _phi_chis.append(_phi_chi)

        _phi_chis   = np.array(_phi_chis)
        _chi_V_phis = np.array(_chi_V_phis)


        _num_of_inputs = np.sum(self.factors)
        if kind == 'bell':
            _fidelity = np.abs(np.sum(_phi_chis[:, -1]))**2/_num_of_inputs**2
            _gradient = 2 * np.imag(np.einsum('qit,pt->it', _chi_V_phis, _phi_chis)) / _num_of_inputs**2


        elif kind == 'average':
            _fidelity = np.abs(np.sum(_phi_chis[:, -1]))**2 + \
                        np.sum(np.abs(_phi_chis[:, -1]/self.factors)**2 * self.factors)\
                        / _num_of_inputs * (_num_of_inputs + 1)
            _gradient = 2 * (np.imag(np.einsum('qit,pt->it', _chi_V_phis, _phi_chis)) +
                             np.imag(np.einsum('qit,qt->it', _chi_V_phis, _phi_chis)))\
                        / _num_of_inputs * (_num_of_inputs + 1)
        else:
            raise ValueError('The kind of fidelity was not specified correctly')

        if symmetric:
            _gradient = (_gradient - _gradient[:, ::-1])/2
        dt = self.times[1] - self.times[0]
        self.fidelity = _fidelity
        self.gradient = _gradient*dt

        if self.freq_cutoff is not None:
            for ii in range(len(self.gradient)):
                freqs = np.fft.rfftfreq(len(self.gradient[ii]), d=dt)
                fft_data = np.fft.rfft(self.gradient[ii])
                fft_data[freqs > self.freq_cutoff/2/pi] = 0
                self.gradient[ii] = np.fft.irfft(fft_data)
        return _fidelity, _gradient


    def step_BFGS(self, _parameters=None):
        _parameters = _parameters.reshape(len(self.operators[0]), -1)
        self.opt_parameters = _parameters
        self.update_pulses()
        self.generate_phis()
        self.generate_chis()
        self.fidelity_calculator(parameters=_parameters)
        return self.fidelity, self.gradient

    def BFGS(self, maxiter=20, ftol=1e-9, disp=False):
        if self.status != 'configured':
            raise RuntimeError('The optimizer is not configured yet')

        self.status = 'running'
        self.opt_parameters = np.array(self.opt_parameters).flatten()
        self.iterations = 0
        pulse_history    = []
        fidelity_history = []

        if disp:
            plt.ion()
            fig = plt.figure(figsize=(12, 4), dpi=100)
            gs_main = GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

            ax = fig.add_subplot(gs_main[0])

            gs_sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1], hspace=0.3)

            ax2_1 = fig.add_subplot(gs_sub[0, 0])
            ax2_2 = fig.add_subplot(gs_sub[1, 0])
            ax2_3 = fig.add_subplot(gs_sub[2, 0])
            # ax_inset = inset_axes(ax,
            #                       width="100%", height="100%",
            #                       loc="lower left",
            #                       bbox_to_anchor=(0.11, 0.58, 0.38, 0.38),
            #                       bbox_transform=ax.transAxes,
            #                       borderpad=0)
            # ax_inset.set_xlabel("Iteration", fontsize=8)
            # ax_inset.set_ylabel("$E = 1 - F$", fontsize=8)
            # ax_inset.tick_params(axis='both', which='major', labelsize=6)
            # ax_inset.set_yscale('log')


        def objective(parameters):
            return 1 - self.step_BFGS(parameters)[0]

        def jacob(_):
            return -self.gradient.flatten()

        def callback(_):
            self.iterations += 1
            fidelity = self.fidelity
            fidelity_history.append(1 - fidelity)

            current_pulse = self.systems[0].pulses[1](self.times)
            pulse_history.append(current_pulse)

            if disp:
                ax.cla()

                for ii, pulse in enumerate(pulse_history[-6::]):
                    alpha = (ii + 1)**1.8 / len(pulse_history[-6::])**1.8
                    ax.plot(self.times, pulse, color='red', alpha=alpha, lw=1)
                    # ax2.plot(self.times, pulse+pulse[::-1], '--r', alpha=alpha, lw=1)
                # ax2.plot(self.times, self.pulses[0](self.times), color='darkblue', alpha=0.8, lw=1)

                ax2_1.cla()
                ax2_2.cla()
                ax2_3.cla()




                ax2_1.plot(self.times, np.abs(self.phis[2][:, 0])**2, 'k', alpha=0.7, lw=1.1)
                ax2_1.plot(self.times, np.abs(self.phis[2][:, 3])**2, 'g', alpha=0.7, lw=1.1)

                ax2_2.plot(self.times, np.abs(self.phis[3][:, 0])**2, 'k', alpha=0.7, lw=1.1)
                ax2_2.plot(self.times, np.abs(self.phis[3][:, 1])**2 +
                                       np.abs(self.phis[3][:, 2])**2, 'g', alpha=0.7, lw=1.1)
                ax2_2.plot(self.times, np.abs(self.phis[3][:, 3])**2, '--r', alpha=0.7, lw=1.1)

                ax2_3.plot(self.times, np.abs(self.phis[4][:, 0])**2, 'k', alpha=0.7, lw=1.1)
                ax2_3.plot(self.times, np.abs(self.phis[4][:, int('011', 2)])**2, 'g', alpha=0.7, lw=1.1)
                ax2_3.plot(self.times, np.abs(self.phis[4][:, int('101', 2)])**2 +
                                       np.abs(self.phis[4][:, int('110', 2)])**2, '--r', alpha=0.7, lw=1.1)


                ax.set_xlabel("Time", fontsize=10, labelpad=5)
                ax.set_ylabel("Detuning", fontsize=10, labelpad=5)
                ax.set_ylim(-200, 200)
                ax.tick_params(axis='both', which='major', labelsize=8)

                ax.set_xlim(self.times[0], self.times[-1])
                # y_min, y_max = min(pulse_history[0]), max(pulse_history[0])
                # y_padding = 0.1 * (y_max - y_min)
                # ax.set_ylim(y_min -  y_padding, y_max + 8*y_padding)

                # ax_inset.cla()
                # ax_inset.step(range(1, len(fidelity_history) + 1), fidelity_history, where='mid', color='blue', lw=1)
                # ax_inset.set_xlim(1, max(len(fidelity_history), 2))
                # ax_inset.set_yscale('log')
                # ax_inset.set_ylim(1e-9, 1)
                # ax_inset.set_xlabel("Iterations", fontsize=6)
                # ax_inset.set_ylabel("Infidelity", fontsize=6)

                plt.draw()
                plt.pause(0.01)

        minimize(fun     =objective,
                 x0      =self.opt_parameters,
                 method  ='L-BFGS-B',
                 jac     =jacob,
                 tol     =ftol,
                 callback=callback,
                 options ={'disp': True, 'maxiter': maxiter})

        if disp:
            plt.ioff()
            plt.show()






if __name__ == '__main__':
    pass

