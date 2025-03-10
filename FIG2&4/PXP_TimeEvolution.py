from QMClasses import*
import os
np.set_printoptions(precision=6, suppress=False, floatmode='maxprec')

B      = 45 * 2*pi
Omega0 = 8  * 2*pi
Delta  = 20  * 2*pi
T      = 2
dev    = 1/2.6 * T/2

TIMES     = []
HISTORIES = []

os.makedirs("Data_PXP", exist_ok=True)

print('n', 'Population Loss', 'Phase', sep='\t\t')
print('------------------------------------------')
for nn in [3, 4, 5]:
    system = Hamiltonians(num_of_atoms=nn)
    system.chain()
    system.pxp_sw_nnn(truncate_rr=False)
    system.double_chirped_linear(t=T, delta=Delta, amplitude=Omega0, dev=dev)
    input_state = np.eye(2**nn, dtype=np.complex128)[0]

    np.save('Data_PXP/omega.npy', system.pulses[0](system.times))
    np.save('Data_PXP/detuning.npy', system.pulses[1](system.times))
    np.save('Data_PXP/times0.npy', system.times)

    simulator = QuantumSimulator()
    simulator.times = system.times
    simulator.f_propagate(input_state=input_state, hamiltonian=system.hamiltonian, rtol=1e-11, atol=1e-11)
    TIMES.append(simulator.times)
    HISTORIES.append(simulator.history)

    G_population = 1 - np.abs(simulator.history[-1, 0])**2
    G_angle      = np.angle(simulator.history[-1, 0])
    print(nn, f"{G_population:.6e}", f"{G_angle:.6e}", sep='\t\t')




np.save('Data_PXP/times3.npy', TIMES[0])
np.save('Data_PXP/times4.npy', TIMES[1])
np.save('Data_PXP/times5.npy', TIMES[2])

np.save('Data_PXP/history3.npy', HISTORIES[0])
np.save('Data_PXP/history4.npy', HISTORIES[1])
np.save('Data_PXP/history5.npy', HISTORIES[2])
print('------------------------------------------')
