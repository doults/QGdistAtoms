import os
from QMClasses import *

Gamma  = 0.0005 * 2*pi
# Gamma  = 0.
B      = 45 * 2*pi
Omega0 = 8  * 2*pi
Delta  = 20 * 2*pi

Duration = np.linspace(start=1., stop=4., num=21)

for N in [3, 4, 5, 6, 7, 8]:
    Errors  = []
    for ii, duration in enumerate(Duration):
        Gate = np.zeros((4, 4), dtype=np.complex128)
        Outputs = []
        for nn in [N-2, N-1, N]:
            input_state = np.eye(2**nn, dtype=np.complex128)[0]
            system      = Hamiltonians(num_of_atoms=nn)
            system.chain()
            system.rydberg_3D_array_vdw(block=B, decay_r=Gamma, flip_b=True)
            system.double_chirped_linear(delta=Delta, t=duration, dev=duration/2/2.6, amplitude=Omega0)

            simulator = QuantumSimulator()
            simulator.times = system.times
            simulator.f_propagate(input_state=input_state, hamiltonian=system.hamiltonian, atol=1e-8, rtol=1e-8)

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
        print('i =', ii)
        print('B/2π =', B/2/pi)
        print('Ω/2π =', Omega0/2/pi)
        print('Δ/2π =', Delta/2/pi)
        print('duration =', duration)
        print('error    =', error)

    save_path = f'Data_tot2/N{N}/decay{Gamma:.5f}_block{B:.2f}_omega{Omega0:.3f}_delta{Delta:.3f}'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + f'/errors', np.array(Errors))
    np.save(save_path + f'/durations', np.array(Duration))