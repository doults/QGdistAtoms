from QMOptimize import *
import os

N = 5
B       = 40  * 2*pi
Omega_0 = 8   * 2*pi
Delta_0 = 16  * 2*pi
Gamma   = 0.5 * 2*pi * 1e-3

system3 = Hamiltonians(num_of_atoms=3)
system4 = Hamiltonians(num_of_atoms=4)
system5 = Hamiltonians(num_of_atoms=5)

in_state3 = np.eye(2**3, dtype=np.complex128)[0]
in_state4 = np.eye(2**4, dtype=np.complex128)[0]
in_state5 = np.eye(2**5, dtype=np.complex128)[0]

systems    = [system3, system4, system5]
in_states  = [in_state3, in_state4,  in_state5]
tar_states = [in_state3, in_state4, -in_state5]



current_dir = os.getcwd()
load_dir = current_dir + '/Data_N5/improved/B40_Omega8_Delta16/'

Fidelities = []
for directory in [name for name in sorted(os.listdir(load_dir))]:
    area     = float(directory[4:])
    duration = area / Omega_0 / 0.75270854
    opt_par = np.load(load_dir + directory + '/Opt_par.npy')

    for syst in systems:
        syst.chain()
        syst.rydberg_3D_array_vdw(block=B, decay_r=Gamma, power=6, flip_b=True)
        syst.double_chirped_linear(t=duration, amplitude=Omega_0, delta=Delta_0, dev=duration/2/2.5, power=8)

    simulator = Optimizer()
    simulator.configure(input_states =in_states,
                        target_states=tar_states,
                        systems      =systems,
                        initial_guess=opt_par,
                        factors      =[1, 2, 1],
                        rk_pre       =1e-8,
                        resolution   =500)
    simulator.update_pulses()
    simulator.generate_phis()
    simulator.generate_chis()

    fid, _ = simulator.fidelity_calculator(parameters=opt_par, kind='average')
    Fidelities.append(fid)
    print(area, 1 - fid)

save_dir = current_dir +  f'/Data_N{N}/decay/B40_Omega8_Delta16'
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + f'/fidelities.npy', np.array(Fidelities))
