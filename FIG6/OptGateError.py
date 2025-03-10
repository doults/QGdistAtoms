import os
from QMClasses import *
from QMFunctions import ckz_gate_error


Gamma  = 1
Bs     = np.logspace(4.4, 6.6, 7)
ratio1 = 0.1778
ratio2 = 0.4444
Omegas = Bs * ratio1
Deltas = Bs * ratio2

Mus = [1, 2, 1, 2, 1, 2]
Cnus = [0.431, 0.432, 0.281, 0.284, 0.189, 0.199]

for N, mu, c_nu in zip([3, 4, 5, 6, 7, 8], Mus, Cnus):
    nu_r = N/2 - 0.25
    Errors = []

    for Omega0, Delta, B in zip(Omegas, Deltas, Bs):
        Topt = 1/c_nu / (Omega0**2 / Delta) * np.log(mu * c_nu/nu_r * (Omega0**2 / Delta) / Gamma)

        Gate = np.zeros((4, 4), dtype=np.complex128)
        Outputs = []
        for nn in [N-2, N-1, N]:
            system = Hamiltonians(num_of_atoms=nn)
            system.chain()
            system.rydberg_3D_array_vdw(block=B, decay_r=Gamma, flip_b=True)
            input_state = np.eye(2**nn, dtype=np.complex128)[0]
            system.double_chirped_linear(delta=Delta, t=2*Topt, dev=Topt/2.6, amplitude=Omega0)

            simulator = QuantumSimulator()
            simulator.times = system.times
            simulator.f_propagate(input_state=input_state, hamiltonian=system.hamiltonian, rtol=1e-9, atol=1e-9)
            output = simulator.history[-1, 0]
            Outputs.append(output)

        Gate[0, 0] = Outputs[0]
        Gate[1, 1] = Outputs[1]
        Gate[2, 2] = Outputs[1]
        Gate[3, 3] = Outputs[2]
        if N%2 == 0: CANONICAL = False
        else:        CANONICAL = True
        error = ckz_gate_error(Gate, canonical=CANONICAL)
        Errors.append(error)

        print('----------------------------------------------')
        print('----------------------------------------------')
        print('N =', N)
        print('B/2π =', B/2/pi)
        print('Ω/2π =', Omega0/2/pi)
        print('Δ/2π =', Delta/2/pi)
        print('duration = ', Topt)
        print('error    = ', error)

    save_path = f'Data_opt/N{N}/ratio1{ratio1}_ratio2{ratio2}'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + f'/errors', np.array(Errors))
    np.save(save_path + f'/B_div_Gamma', np.array(Bs / Gamma))