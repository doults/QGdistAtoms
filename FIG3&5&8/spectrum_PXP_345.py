import os
from QMClasses import*
from QMFunctions import*


Omega0 = 8 * 2*pi
Delta  = 3*Omega0
T      = 1
dev    = 1/3 * T


################################# N=3 #######################################
system3 = Hamiltonians(num_of_atoms=3)
system3.chain()
system3.pxp_sw_nnn()
system3.chirped_linear(delta    =Delta,
                       amplitude=Omega0,
                       t        =T,
                       dev      =T/2.5,
                       omega_const=True)

simulator = QuantumSimulator()
simulator.times = system3.times

vals, vecs = simulator.spectrum(hamiltonian=system3.hamiltonian)     # Calculate E_k(t), |α_k(t)>
vecs     = np.einsum('tim, tm->tim', vecs, np.sign(vecs[:, 0, :]))   # Fix the phase of |α_k(t)> for all t
dt       = simulator.times[1] - simulator.times[0]
der_vecs = np.gradient(vecs, dt, axis=0)                             # Calculate |d α_k(t)/dt>

detuning = system3.pulses[1](system3.times)
amp1 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 0], np.conj(der_vecs))) ** 2 / Delta
amp2 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 4], np.conj(der_vecs)))**2/Delta

os.makedirs('Data_PXP/N3', exist_ok=True)
np.save('Data_PXP/N3/detuning.npy', detuning)
np.save('Data_PXP/N3/vals.npy', vals)
np.save('Data_PXP/N3/vecs.npy', vecs)
np.save('Data_PXP/N3/amp1.npy', amp1)
np.save('Data_PXP/N3/amp2.npy', amp2)


################################# N=4 #######################################
system4 = Hamiltonians(num_of_atoms=4)
system4.chain()
system4.pxp_sw_nnn()
system4.chirped_linear(delta    =Delta,
                       amplitude=Omega0,
                       t        =T,
                       dev      =T/2.5,
                       omega_const=True)

simulator = QuantumSimulator()
simulator.times = system4.times
vals, vecs = simulator.spectrum(hamiltonian=system4.hamiltonian)     # Calculate E_k(t), |α_k(t)>
vecs     = np.einsum('tim, tm->tim', vecs, np.sign(vecs[:, 0, :]))   # Fix the phase of |α_k(t)> for all t

# Fix the enumeration of eigenstates, eigenvalues for all t
vals[system4.times > T/2, 3], vals[system4.times > T/2, 4] = vals[system4.times > T/2, 4], vals[system4.times > T/2, 3]
vecs[system4.times > T/2, :, 3], vecs[system4.times > T/2, :, 4] = vecs[system4.times > T/2, :, 4], vecs[system4.times > T/2, :, 3]
dt       = simulator.times[1] - simulator.times[0]
der_vecs = np.gradient(vecs, dt, axis=0)                            # Calculate |d α_k(t)/dt>

detuning = system4.pulses[1](system4.times)
amp3 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 0], np.conj(der_vecs)))**2/Delta
amp4 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 7], np.conj(der_vecs)))**2/Delta

os.makedirs('Data_PXP/N4', exist_ok=True)
np.save('Data_PXP/N4/detuning.npy', detuning)
np.save('Data_PXP/N4/vals.npy', vals)
np.save('Data_PXP/N4/vecs.npy', vecs)
np.save('Data_PXP/N4/amp3.npy', amp3)
np.save('Data_PXP/N4/amp4.npy', amp4)


################################# N=5 #######################################
system5 = Hamiltonians(num_of_atoms=5)
system5.chain()
system5.pxp_sw_nnn()
system5.chirped_linear(delta    =Delta,
                       amplitude=Omega0,
                       t        =T,
                       dev      =T/2.5,
                       omega_const=True)

simulator = QuantumSimulator()
simulator.times = system5.times
vals, vecs = simulator.spectrum(hamiltonian=system5.hamiltonian)     # Calculate E_k(t), |α_k(t)>
vecs     = np.einsum('tim, tm->tim', vecs, np.sign(vecs[:, 0, :]))   # Fix the phase of |α_k(t)> for all t

# Fix the enumeration of eigenstates, eigenvalues for all t
der_vecs = np.gradient(vecs, dt, axis=0)
raw_amp = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 0], np.conj(der_vecs))) ** 2 / Delta
con45 = raw_amp[:, 4] > raw_amp[:, 5]
con78 = raw_amp[:, 7] > raw_amp[:, 8]
vals[con45, 4], vals[con45, 5] = vals[con45, 5], vals[con45, 4]
vals[con78, 7], vals[con78, 8] = vals[con78, 8], vals[con78, 7]
vecs[con45, :, 4], vecs[con45, :, 5] = vecs[con45, :, 5], vecs[con45, :, 4]
vecs[con78, :, 7], vecs[con78, :, 8] = vecs[con78, :, 8], vecs[con78, :, 7]
dt       = system5.times[1] - system5.times[0]
der_vecs = np.gradient(vecs, dt, axis=0)                             # Calculate |d α_k(t)/dt>

detuning = system5.pulses[1](system3.times)
amp5 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 0], np.conj(der_vecs)))**2/Delta
amp6 = np.abs(np.einsum('ti,tim->tm', vecs[:, :, 12], np.conj(der_vecs)))**2/Delta

os.makedirs('Data_PXP/N5', exist_ok=True)
np.save('Data_PXP/N5/detuning.npy', detuning)
np.save('Data_PXP/N5/vals.npy', vals)
np.save('Data_PXP/N5/vecs.npy', vecs)
np.save('Data_PXP/N5/amp5.npy', amp5)
np.save('Data_PXP/N5/amp6.npy', amp6)