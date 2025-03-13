from QMOptimize import *
import os


N = 5
sample_size = 1

B       = 40 * 2*pi
Omega_0 = 8  * 2*pi
Delta_0 = 16 * 2*pi
variance = 0.01
# variance = 0.


in_state3 = np.eye(2**3, dtype=np.complex128)[0]
in_state4 = np.eye(2**4, dtype=np.complex128)[0]
in_state5 = np.eye(2**5, dtype=np.complex128)[0]

system3  = Hamiltonians(num_of_atoms=3)
system4r = Hamiltonians(num_of_atoms=4)
system4l = Hamiltonians(num_of_atoms=4)
system5  = Hamiltonians(num_of_atoms=5)

systems    = [system3, system4r, system4l, system5]
in_states  = [in_state3, in_state4, in_state4,  in_state5]
tar_states = [in_state3, in_state4, in_state4, -in_state5]





AREAS   = []
FID_all = []
HIST_00 = []
HIST_01 = []
HIST_10 = []
HIST_11 = []
TIMES   = []


current_dir = os.getcwd()
data_dir    = current_dir + f'/Data_N{N}/improved/B40_Omega8_Delta16/'


print(f'{"Area":<12} {"τ":<12} {"Error":<20}')
print('-----------------------------------------')

for directory in [name for name in sorted(os.listdir(data_dir))][::-1]:
    area    = float(directory[4:])
    T       = area / Omega_0 / 0.75270854
    opt_par = np.load(data_dir + directory + '/Opt_par.npy')
    FID = []
    for ii in range(sample_size):
        displacements = np.random.normal(loc=0, scale=variance, size=(5, 3))

        system3.chain()
        system3.positions += displacements[1:-1]
        system3.rydberg_3D_array_vdw(block=B, decay_r=0, power=6, flip_b=True)
        system3.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2/2.5, power=8)

        system4r.chain()
        system4r.positions += displacements[1:]
        system4r.rydberg_3D_array_vdw(block=B, decay_r=0, power=6, flip_b=True)
        system4r.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2/2.5, power=8)

        system4l.chain()
        system4l.positions += displacements[:-1]
        system4l.rydberg_3D_array_vdw(block=B, decay_r=0, power=6, flip_b=True)
        system4l.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2/2.5, power=8)

        system5.chain()
        system5.positions += displacements
        system5.rydberg_3D_array_vdw(block=B, decay_r=0, power=6, flip_b=True)
        system5.double_chirped_linear(t=T, amplitude=Omega_0, delta=Delta_0, dev=T/2/2.5, power=8)


        simulator = Optimizer()
        simulator.configure(input_states =in_states,
                            target_states=tar_states,
                            systems      =systems,
                            initial_guess=opt_par,
                            factors      =[1, 1, 1, 1],
                            rk_pre       =1e-8,
                            resolution   =500)
        simulator.update_pulses()
        simulator.generate_phis()
        simulator.generate_chis()

        fid, _ = simulator.fidelity_calculator(parameters=opt_par, kind='bell')
        FID.append(fid)
        HIST_00.append(simulator.phis[0])
        HIST_01.append(simulator.phis[1])
        HIST_10.append(simulator.phis[1])

        HIST_11.append(simulator.phis[3])
        TIMES.append(simulator.times)
    FID = np.array(FID)

    print(f'{area/2:<12.4} {T/2:<12.4} {np.average(1 - FID):<8.6f} ± {np.var(1 - FID):.6f}')
    FID_all.append(np.array(FID))
FID_all = np.array(FID_all)

save_dir = current_dir + '/Data_N5/disordered/B40_Omega8_Delta16' + f'/variance{variance}'
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + '/fidelities.npy', np.array(FID_all))


np.save(save_dir + '/histories_00.npy', HIST_00)
np.save(save_dir + '/histories_01.npy', HIST_01)
np.save(save_dir + '/histories_10.npy', HIST_10)
np.save(save_dir + '/histories_11.npy', HIST_11)
np.save(save_dir + '/times.npy', TIMES)