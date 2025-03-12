import os
from QMOptimize import *


N = 5
B       = 40 * 2*pi
Omega_0 = 8  * 2*pi
Delta_0 = 16 * 2*pi


system3 = Hamiltonians(num_of_atoms=3)
system4 = Hamiltonians(num_of_atoms=4)
system5 = Hamiltonians(num_of_atoms=5)

in_state3 = np.eye(2**3, dtype=np.complex128)[0]
in_state4 = np.eye(2**4, dtype=np.complex128)[0]
in_state5 = np.eye(2**5, dtype=np.complex128)[0]

systems    = [system3, system4, system5]
in_states  = [in_state3, in_state4,  in_state5]
tar_states = [in_state3, in_state4, -in_state5]

OPT_PAR = []
FID     = []

Tlist = list(np.linspace(1., 0.8, 41))
for ii, T in enumerate(Tlist):
    print(f'-------------------- {ii} : {T} : {round(T * Omega_0 * 0.7527, 4)} --------------------')
    area = np.round(T*Omega_0*0.7527, decimals=4)
    in_guess = np.load(f'Data_N{N}/raw/B40_Omega8_Delta16/Area{area}/Opt_par.npy')
    for syst in systems:
        syst.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2.5/2)
        syst.chain()
        syst.rydberg_3D_array_vdw(block=B, decay_r=0, flip_b=True)

    optimizer = Optimizer()
    optimizer.configure(systems      =systems,
                        input_states =in_states,
                        target_states=tar_states,
                        initial_guess=in_guess,
                        freq_cutoff  =B*2,
                        resolution   =500,
                        rk_pre       =1e-11,
                        factors      =[1, 2, 1])
    optimizer.BFGS(maxiter=150, ftol=1e-10, disp=False)

    directory  = f'Data_N{N}/improved'
    directory += f'/B{round(B/2/pi)}_Omega{round(Omega_0/2/pi)}_Delta{round(Delta_0 /2/pi)}'
    directory += f'/Area{round(T * Omega_0 * 0.7527, 4)}/'
    os.makedirs(directory, exist_ok=True)
    np.save(directory + 'Opt_par.npy', optimizer.opt_parameters)
    np.save(directory + 'Fidelity.npy', optimizer.fidelity)
