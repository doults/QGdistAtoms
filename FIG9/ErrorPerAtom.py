import os

from QMClasses import*
from QMFunctions import*


N = 5

Block    = 40 * 2*pi
B2 = Block/64
Omega_sd = 8 * 2 * pi * np.exp(1j * (pi / 2))


proj_single = np.outer(np.eye(3)[2], np.eye(3)[2])
proj_edge   = np.kron(proj_single, np.eye(3 ** (N - 1)))
proj_bulk   = np.kron(np.eye(3 ** ((N - 1) // 2)), np.kron(proj_single, np.eye(3 ** ((N - 1) // 2))))


Deltas_sd = np.linspace(0, -3.5 * B2, 36)
Bulk = []
Edge = []
print(f'{"-δ_SD/B2":<15} {"E_edge":<15} {"E_bulk":<15}')
print('-' * 45)
for ii, delta_sd in enumerate(Deltas_sd):
    input_state = np.eye(3**N, dtype=np.complex128)[int('10101', 3)]

    system = Hamiltonians(num_of_atoms=N)
    system.chirped_pi_chirped(dur     =pi / np.abs(Omega_sd) + 1e-3,
                              omega_gr=0,
                              omega_ar=Omega_sd,
                              delta_gr=0,
                              delta_aa=delta_sd)
    system.chain()
    system.rydberg_3D_array_vdw_3l(block=Block, block_cross=0, power=6)

    simulator = QuantumSimulator()
    simulator.times = system.times
    simulator.f_propagate(input_state=input_state, hamiltonian=system.hamiltonian, rtol=1e-7, atol=1e-7)


    edge = np.abs(np.einsum('ti,ij,tj->t', simulator.history, proj_edge, simulator.history.conj()))
    bulk = np.abs(np.einsum('ti,ij,tj->t', simulator.history, proj_bulk, simulator.history.conj()))

    Edge.append(edge[-1].real)
    Bulk.append(bulk[-1].real)
    print(f'{-delta_sd / B2:<15.1f} {1 - edge[-1].real:<15.6f} {1 - bulk[-1].real:<15.6f}')

print('-' * 45)

os.makedirs('Data', exist_ok=True)
np.save('Data/delta_sd_B2.npy', np.array(Deltas_sd/B2))
np.save('Data/Edge.npy', np.array(Edge))
np.save('Data/Bulk.npy', np.array(Bulk))