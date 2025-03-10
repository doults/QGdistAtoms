import string
from itertools  import combinations
from matplotlib import pyplot as plt
import scipy.integrate as inte
from QMFunctions import*

np.set_printoptions(linewidth=200)

h_bar = 1
light_speed = 1


class Tools(object):
    """This class contains functions for manipulating the hilbert space, as well
    basic operators for different systems"""

    def __init__(self, num_of_atoms):
        super().__init__()
        self.num_of_atoms = num_of_atoms
        self.hil_dim      = 2**num_of_atoms
        self.basis = np.eye(2, dtype=np.complex128)


    def __expand__(self, matrix, from_subspace):
        """This function expands a matrix that lives in a hilbert subspace
        indexed with integer 'from_subspace', to the full hilbert space with
        dimension 'full_hil_dim' and has the form:

        A_plus* = I x I x ... x A_i x ... I
        """

        if from_subspace not in range(self.num_of_atoms):
            raise ValueError
        identity = np.eye(len(matrix), dtype=np.complex128)

        for _ in range(from_subspace):
            matrix = np.kron(identity, matrix)
        for _ in range(self.num_of_atoms - from_subspace - 1):
            matrix = np.kron(matrix, identity)

        return matrix


    def __cross_pow__(self, matrix, exponent=None):
        """This function applies the same matrix to each subspace."""
        if exponent is None:
            exponent = self.num_of_atoms
        output = matrix

        if exponent >= 1:
            for _ in range(exponent - 1):
                output = np.kron(matrix, output)
            return output
        else:
            return None


    def __pair_expand__(self, matrix1, matrix2, from_subspaces=(0, 1)):
        """This method expands to matrices A_plus and B so that outputs dimensionality
        covers the full Hilbert space. The output has the form:
        A_i x B_j = I x ... I x A x I x ... I x B x I .... I
        """
        if from_subspaces[0] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[1] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[0] >= from_subspaces[1]           : raise ValueError

        eye_1 = self.__cross_pow__(self.basis, exponent=from_subspaces[0])
        eye_2 = self.__cross_pow__(self.basis, exponent=from_subspaces[1] - from_subspaces[0] - 1)
        eye_3 = self.__cross_pow__(self.basis, exponent=self.num_of_atoms - from_subspaces[1] - 1)

        output = matrix1
        if eye_1 is not None: output = np.kron(eye_1, output)
        if eye_2 is not None: output = np.kron(output, eye_2)
        output = np.kron(output, matrix2)
        if eye_3 is not None: output = np.kron(output, eye_3)

        return output


    def __triplet_expand__(self, matrix1, matrix2, matrix3, from_subspaces=(0, 1, 2)):
        """This method expands to matrices A_plus and B so that outputs dimensionality
        covers the full Hilbert space. The output has the form:
        A_i x B_j = I x ... I x A x I x ... I x B x I .... I
        """
        if from_subspaces[0] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[1] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[2] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[0] >= from_subspaces[1]           : raise ValueError
        if from_subspaces[1] >= from_subspaces[2]           : raise ValueError


        eye_1 = self.__cross_pow__(self.basis, exponent=from_subspaces[0])
        eye_2 = self.__cross_pow__(self.basis, exponent=from_subspaces[1] - from_subspaces[0] - 1)

        eye_3 = self.__cross_pow__(self.basis, exponent=from_subspaces[2] - from_subspaces[1] - 1)
        eye_4 = self.__cross_pow__(self.basis, exponent=self.num_of_atoms - from_subspaces[2] - 1)


        output = matrix1
        if eye_1 is not None: output = np.kron(eye_1, output)
        if eye_2 is not None: output = np.kron(output, eye_2)
        output = np.kron(output, matrix2)
        if eye_3 is not None: output = np.kron(output, eye_3)
        output = np.kron(output, matrix3)
        if eye_4 is not None: output = np.kron(output, eye_4)
        return output


    def __quad_expand__(self, matrix1, matrix2, matrix3, matrix4, from_subspaces=(0, 1, 2, 3)):
        """This method expands to matrices A_plus and B so that outputs dimensionality
        covers the full Hilbert space. The output has the form:
        A_i x B_j x C_k x D_l= I x ... I x A x I x ... I x B x I .... I x C x I .... I x D x I ....I
        """
        if from_subspaces[0] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[1] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[2] not in range(self.num_of_atoms): raise ValueError
        if from_subspaces[3] not in range(self.num_of_atoms): raise ValueError

        if from_subspaces[0] >= from_subspaces[1]           : raise ValueError
        if from_subspaces[1] >= from_subspaces[2]           : raise ValueError
        if from_subspaces[2] >= from_subspaces[3]           : raise ValueError


        eye_1 = self.__cross_pow__(self.basis, exponent=from_subspaces[0])
        eye_2 = self.__cross_pow__(self.basis, exponent=from_subspaces[1] - from_subspaces[0] - 1)
        eye_3 = self.__cross_pow__(self.basis, exponent=from_subspaces[2] - from_subspaces[1] - 1)
        eye_4 = self.__cross_pow__(self.basis, exponent=from_subspaces[3] - from_subspaces[2] - 1)
        eye_5 = self.__cross_pow__(self.basis, exponent=self.num_of_atoms - from_subspaces[3] - 1)


        output = matrix1
        if eye_1 is not None: output = np.kron(eye_1, output)
        if eye_2 is not None: output = np.kron(output, eye_2)
        output = np.kron(output, matrix2)
        if eye_3 is not None: output = np.kron(output, eye_3)
        output = np.kron(output, matrix3)
        if eye_4 is not None: output = np.kron(output, eye_4)
        output = np.kron(output, matrix4)
        if eye_5 is not None: output = np.kron(output, eye_5)
        return output


    def __op_ket__(self, operator, ket):
        letters = string.ascii_lowercase[0:self.num_of_atoms*2]
        letters += ','  + letters[1::2] + '->' + letters[::2]
        return np.einsum(letters, operator, ket)


class Geometry(object):
    """
    A class to define and visualize the geometric arrangement of neutral atoms.

    Attributes:
    -----------
    num_of_atoms : int
        The number of atoms in the geometry.
    positions : np.ndarray
        An array of shape (N, 3) representing the positions of the atoms in 3D space.
    qubit_idx : list[int]
        Indices of the qubits in the geometry.
    """

    def __init__(self, num_of_atoms):
        """
        Parameters:
        -----------
        num_of_atoms : int
            Number of atoms in the geometry.
        """
        self.num_of_atoms = num_of_atoms
        self.positions = None
        self.qubit_idx = None

    def plot_geometry(self):
        """
        Plot the 3D arrangement of atoms, with qubits highlighted in red.
        """
        if self.positions is None:
            raise ValueError("Positions are not defined. Call a geometry method to define positions first.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xx, yy, zz = self.positions.T

        max_coord = np.max(np.linalg.norm(self.positions, axis=1))
        aux_pos = [i for i in range(len(self.positions)) if i not in self.qubit_idx]

        ax.scatter(xx[aux_pos], yy[aux_pos], zz[aux_pos], s=250 / self.num_of_atoms, c='k', marker='o', label='Atoms')
        ax.scatter(xx[self.qubit_idx], yy[self.qubit_idx], zz[self.qubit_idx], s=250 / self.num_of_atoms, c='r', marker='o', label='Qubits')

        for i, (x, y, z) in enumerate(zip(xx, yy, zz)):
            ax.text(x, y, z + 0.3, s=f'{i}')

        ax.set_xlim(-1.05 * max_coord, 1.05 * max_coord)
        ax.set_ylim(-1.05 * max_coord, 1.05 * max_coord)
        ax.set_zlim(-1.05 * max_coord, 1.05 * max_coord)
        ax.set_xlabel(r'$x/\alpha$')
        ax.set_ylabel(r'$y/\alpha$')
        ax.set_zlabel(r'$z/\alpha$')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def chain(self):
        """
        Arrange atoms in a 1D chain along the x-axis.

        Returns:
        --------
        np.ndarray
            Positions of the atoms.
        """
        self.positions = np.array([[i - (self.num_of_atoms - 1) / 2, 0., 0.] for i in range(self.num_of_atoms)])
        self.qubit_idx = [0, self.num_of_atoms - 1]
        return self.positions

    def star(self):
        """
        Arrange atoms in a star pattern with one central atom and others on a circle.

        Returns:
        --------
        np.ndarray
            Positions of the atoms.
        """
        self.positions = np.array(
            [[0, 0, 0]] + [[np.cos(2 * pi * i / (self.num_of_atoms - 1)), np.sin(2 * pi * i / (self.num_of_atoms - 1)), 0.]
                           for i in range(self.num_of_atoms - 1)]
        )
        self.qubit_idx = list(range(self.num_of_atoms))
        return self.positions

    def star_tree(self, legs):
        """
        Arrange atoms in a star-tree pattern with specified leg lengths.

        Parameters:
        -----------
        legs : list[int]
            List of integers representing the number of atoms on each leg.

        Returns:
        --------
        np.ndarray
            Positions of the atoms.

        Raises:
        -------
        ValueError
            If the total number of atoms does not match the sum of leg lengths plus the central atom.
        """
        if 1 + sum(legs) != self.num_of_atoms:
            raise ValueError('The number of atoms is not compatible with the legs: ')
        positions = [[0, 0, 0]]
        n_legs = len(legs)
        for ii in range(n_legs):
            for jj in range(1, legs[ii]+1):
                positions.append([np.cos(2 * pi * ii / n_legs)*jj, np.sin(2 * pi * ii / n_legs)*jj, 0.])
        self.positions = np.array(positions)
        self.qubit_idx = np.array([sum(legs[:ii]) for ii in range(len(legs) + 1)])
        return self.positions

    def remove_qubit(self, idx):
        """
        Remove a qubit at a specified index.

        Parameters:
        -----------
        idx : int
            Index of the qubit to remove.

        Returns:
        --------
        np.ndarray
            Updated positions of the atoms.
        """
        self.positions = np.delete(self.positions, idx, axis=0)
        self.qubit_idx = self.qubit_idx[self.qubit_idx != idx]
        self.qubit_idx[self.qubit_idx > idx] -= 1
        return self.positions

    def remove_many(self, key):
        """
        Remove multiple qubits based on a binary key.

        Parameters:
        -----------
        key : str
            Binary string where '1' indicates keeping the qubit and '0' indicates removing it.

        Raises:
        -------
        ValueError
            If the length of the key does not match the number of qubits.
        """
        if len(key) != len(self.qubit_idx):
            raise ValueError("Key length must match the number of qubits.")

        for bit, idx in zip(key[::-1], sorted(self.qubit_idx, reverse=True)):
            if bit == '0':
                self.remove_qubit(idx=idx)


class Pulses(object):
    """
    A class for managing and visualizing pulse sequences for laser control on qubits.
    This class provides methods to create pulse sequences including adiabatic and square pulses,
    as well as tools for plotting these pulses.
    """

    def __init__(self):
        """
        Parameters:
        -----------
        num_of_atoms : int
            The number of atoms in the system.
        """
        self.times = None
        self.pulses = None

    def plot_pulses(self):
        if self.times is None or self.pulses is None:
            raise ValueError("Times and pulses must be defined before plotting.")

        plt.figure(figsize=(8 / 2.53, 5 / 2.53))
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        for pulse_function in self.pulses:
            omega = [pulse_function(t) for t in self.times]
            plt.plot(self.times, omega, alpha=0.6)

        plt.tight_layout()
        plt.show()

    def constDelta_linOmega(self, delta, duration, amplitude, steps=100):
        self.times    = np.linspace(0, duration, steps)
        self.pulses = []


        def rabi_fre(time):
            return time/duration*amplitude

        def detuning(time):
            return delta + 0*time

        self.pulses.extend([rabi_fre, detuning])


    def chirped_linear(self, delta, t, dev, amplitude, power=8, omega_const=False):
        self.times = np.linspace(0, t, 1000)
        self.pulses = []

        def rabi_fre(time):
            if omega_const:
                pulse1 = amplitude + 0*time
            else: pulse1 = Omega_gaussian_pow(time, start=0., duration=t, dev=dev, amplitude=amplitude, power=power)
            return pulse1

        def detuning(time):
            return -delta + 2 * delta / t * time

        self.pulses.extend([rabi_fre, detuning])

    def double_chirped_linear(self, delta, t, dev, amplitude, power=8):
        self.times = np.linspace(0, t, 1000)
        self.pulses = []

        def rabi_fre(time):
            pulse1 = Omega_gaussian_pow(time, start=0., duration=t/2, dev=dev, amplitude=amplitude, power=power)
            pulse2 = Omega_gaussian_pow(time, start=t/2, duration=t/2, dev=dev, amplitude=amplitude, power=power)
            return np.where(time <= t/2, pulse1, pulse2)

        def detuning(time):
            return np.where(time <= t / 2,
                            -delta + 2 * delta / (t / 2) * time,
                            -delta + 2 * delta / (t / 2) * (time - t / 2))
        self.pulses.extend([rabi_fre, detuning])

    def chirped_pi_chirped(self, dur, delta_gr, omega_gr, delta_aa, omega_ar):
        dur2 = pi / np.abs(omega_ar)
        dur1 = (dur - dur2) / 2
        dur3 = dur1
        self.times  = np.linspace(0, dur, 1000)
        self.pulses = []

        def rabi_fre_gr(time):
            rabi1 = Omega_gaussian_pow(time, start=0., duration=dur1, dev=dur1/2.5, amplitude=omega_gr, power=8)
            return np.where(time <= dur1, rabi1, 0)
        self.pulses.append(rabi_fre_gr)

        def detuning_gr(time):
            detuning1 = -delta_gr + 2 * delta_gr / dur1 * time
            return np.where(time <= dur1, detuning1, 0)
        self.pulses.append(detuning_gr)

        def rabi_fre_ar(time):
            rabi2 = omega_ar + time*0
            return np.where(time >= dur1, np.where(time <= dur2 + dur1, rabi2, 0), 0)
        self.pulses.append(rabi_fre_ar)

        def detuning_aa(time):
            detuning2 = delta_aa + time*0
            return np.where(time >= dur1, np.where(time <= dur2 + dur1, detuning2, 0), 0)
        self.pulses.append(detuning_aa)


        def rabi_fre_rg(time):
            rabi3 = Omega_gaussian_pow(time, start=dur1 + dur2, duration=dur3, dev=dur3/2.5, amplitude=omega_gr, power=8)
            return np.where(time >= dur1 + dur2, rabi3, 0)
        self.pulses.append(rabi_fre_rg)

        def detuning_rg(time):
            detuning3 = -delta_gr + 2 * delta_gr / dur1 * (time - dur1 - dur2)
            return np.where(time >= dur1 + dur2, detuning3, 0)
        self.pulses.append(detuning_rg)


class Hamiltonians(Geometry, Pulses, Tools):
    """This class contains the dynamics of different quantum systems. It has
    inherited method to create hamiltonians from class Tools"""

    def __init__(self, num_of_atoms):
        super().__init__(num_of_atoms)
        self.hamiltonian = None


    def passive(self):
        h = np.zeros((2**self.num_of_atoms, 2**self.num_of_atoms), dtype=np.complex128)

        def hamiltonian_eff(time):
            return h
        self.hamiltonian = hamiltonian_eff
        return self.hamiltonian

    def rydberg_3D_array_vdw(self, block, decay_r=0, power=6, flip_b=False):
        self.basis = np.eye(2, dtype=np.complex128)
        sigma_rg = np.einsum('i,j->ij', self.basis[1], self.basis[0])
        sigma_rr = np.einsum('i,j->ij', self.basis[1], self.basis[1])

        h_int = 0
        h_rg  = 0
        h_gr  = 0
        h_rr  = 0

        pairs = list(combinations(range(self.num_of_atoms), 2))
        for ii, jj in pairs:
            pos_i = self.positions[ii]
            pos_j = self.positions[jj]
            """∑∑ᵢⱼ Bᵢⱼ|rr><rr|    with     Bᵢⱼ = C₆/||xᵢ - xⱼ||⁶"""
            distance = np.linalg.norm(pos_i - pos_j)

            h_int += block / distance**power * self.__pair_expand__(matrix1=sigma_rr,
                                                                    matrix2=sigma_rr,
                                                                    from_subspaces=(ii, jj))
        for ii in range(self.num_of_atoms):
            """∑ᵢΩ/2|r><1| + h.c - ∑ᵢ(Δ + iΓ/2)|r><r|"""
            h_rg += self.__expand__(sigma_rg,   from_subspace=ii)
            h_gr += self.__expand__(sigma_rg.T, from_subspace=ii)
            h_rr += self.__expand__(sigma_rr,   from_subspace=ii)

        def hamiltonian_eff(time):
            sign_flip = +1
            if flip_b: sign_flip = -np.sign(time - self.times[-1] / 2)
            hamilton = sign_flip * h_int
            hamilton += 1/2 * self.pulses[0](time) * h_rg
            hamilton += 1/2 * np.conj(self.pulses[0](time)) * h_gr
            hamilton -= (self.pulses[1](time) + 1.j * decay_r/2) * h_rr
            return hamilton
        self.hamiltonian = hamiltonian_eff
        return self.hamiltonian


    def pxp_sw_nnn(self, b=1000, nn_neighbors=False, schrieffer_wolf=False, nnn_neighbors=False, power=6,
                   truncate_rr=True, flip_b=False):
        self.basis = np.eye(2, dtype=np.complex128)

        sigma_rg = np.einsum('i,j->ij', self.basis[1], self.basis[0])
        x_pauli = sigma_rg + sigma_rg.T
        p_proj = np.einsum('i,j->ij', self.basis[0], self.basis[0])
        q_proj = np.einsum('i,j->ij', self.basis[1], self.basis[1])


        qs = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        for ii in range(self.num_of_atoms):
            qs +=  self.__expand__(q_proj, from_subspace=ii)

        xp___px = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        if self.num_of_atoms >= 2:
            xp___px += self.__pair_expand__(matrix1=x_pauli,  matrix2=p_proj, from_subspaces=(0, 1))
            xp___px += self.__pair_expand__(matrix1=p_proj,  matrix2=x_pauli, from_subspaces=(self.num_of_atoms - 2,
                                                                                              self.num_of_atoms - 1))

        pxp = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        for ii in range(1, self.num_of_atoms - 1):
            pxp += self.__triplet_expand__(matrix1=p_proj,
                                           matrix2=x_pauli,
                                           matrix3=p_proj, from_subspaces=(ii - 1, ii, ii + 1))

        q_q = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        if nn_neighbors:
            for ii in range(self.num_of_atoms - 2):
                q_q += self.__pair_expand__(matrix1=q_proj, matrix2=q_proj, from_subspaces=(ii, ii + 2))

        q__q = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        if nnn_neighbors:
            for ii in range(self.num_of_atoms - 3):
                q__q += self.__pair_expand__(matrix1=q_proj, matrix2=q_proj, from_subspaces=(ii, ii + 3))

        sw1 = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        sw2 = np.zeros((2 ** self.num_of_atoms, 2 ** self.num_of_atoms), dtype=np.complex128)
        if schrieffer_wolf:
            if self.num_of_atoms > 1:
                sw1 += self.__pair_expand__(matrix1=p_proj, matrix2=q_proj, from_subspaces=(0, 1))
                sw1 += self.__pair_expand__(matrix1=q_proj, matrix2=p_proj, from_subspaces=(self.num_of_atoms - 2,
                                                                                            self.num_of_atoms - 1))

            if self.num_of_atoms == 2:
                """σ+ σ-  +  σ- σ+"""
                sw1 += self.__pair_expand__(matrix1=sigma_rg, matrix2=sigma_rg.T, from_subspaces=(0, 1))
                sw1 += self.__pair_expand__(matrix1=sigma_rg.T, matrix2=sigma_rg, from_subspaces=(0, 1))

            if self.num_of_atoms >= 3:
                for ii in range(1, self.num_of_atoms - 1):
                    """∑ PPQ + QPP + QPQ"""
                    sw1 += self.__triplet_expand__(matrix1=p_proj,
                                                   matrix2=p_proj,
                                                   matrix3=q_proj, from_subspaces=(ii - 1, ii, ii + 1))
                    sw1 += self.__triplet_expand__(matrix1=q_proj,
                                                   matrix2=p_proj,
                                                   matrix3=p_proj, from_subspaces=(ii - 1, ii, ii + 1))
                    sw2 += self.__triplet_expand__(matrix1=q_proj,
                                                   matrix2=p_proj,
                                                   matrix3=q_proj, from_subspaces=(ii - 1, ii, ii + 1))
                """(σ+ σ-  +  σ-σ+)P₃  +  P_{N-2}(σ+ σ-  +  σ- σ+)"""
                sw1 += self.__triplet_expand__(matrix1=sigma_rg,
                                               matrix2=sigma_rg.T,
                                               matrix3=p_proj, from_subspaces=(0, 1, 2))
                sw1 += self.__triplet_expand__(matrix1=sigma_rg.T,
                                               matrix2=sigma_rg,
                                               matrix3=p_proj, from_subspaces=(0, 1, 2))

                sw1 += self.__triplet_expand__(matrix1=p_proj,
                                               matrix2=sigma_rg,
                                               matrix3=sigma_rg.T, from_subspaces=(self.num_of_atoms - 3,
                                                                                   self.num_of_atoms - 2,
                                                                                   self.num_of_atoms - 1))
                sw1 += self.__triplet_expand__(matrix1=p_proj,
                                               matrix2=sigma_rg.T,
                                               matrix3=sigma_rg  , from_subspaces=(self.num_of_atoms - 3,
                                                                                   self.num_of_atoms - 2,
                                                                                   self.num_of_atoms - 1))
            if self.num_of_atoms >= 4:
                for ii in range(1, self.num_of_atoms - 2):
                    sw1 += self.__quad_expand__(matrix1=p_proj,
                                                matrix2=sigma_rg.T,
                                                matrix3=sigma_rg,
                                                matrix4=p_proj, from_subspaces=(ii - 1, ii, ii + 1, ii + 2))
                    sw1 += self.__quad_expand__(matrix1=p_proj,
                                                matrix2=sigma_rg,
                                                matrix3=sigma_rg.T,
                                                matrix4=p_proj, from_subspaces=(ii - 1, ii, ii + 1, ii + 2))

        if truncate_rr:
            truncator = []
            for ii in range(2 ** self.num_of_atoms):
                binary = bin(ii)[2:].zfill(self.num_of_atoms)  # Convert to binary and pad to length 5
                has_adjacent_ones = False
                for jj in range(1, len(binary)):
                    if binary[jj] == '1' and binary[jj - 1] == '1':  # Check for adjacent '1's
                        has_adjacent_ones = True
                        break
                if not has_adjacent_ones:
                    truncator.append(int('0b' + binary, 2))

            qs      = qs[np.ix_(truncator, truncator)]
            xp___px = xp___px[np.ix_(truncator, truncator)]
            pxp     = pxp[np.ix_(truncator, truncator)]
            q_q     = q_q[np.ix_(truncator, truncator)]
            q__q    = q__q[np.ix_(truncator, truncator)]
            sw1     = sw1[np.ix_(truncator, truncator)]
            sw2     = sw2[np.ix_(truncator, truncator)]



        def hamiltonian_eff(time):
            hamilton = 0
            sign_flip = +1
            if flip_b: sign_flip = -np.sign(time - self.times[-1] / 2)
            block = b * sign_flip

            if self.num_of_atoms == 1:
                hamilton += 1 / 2 * self.pulses[0](time) * x_pauli
            hamilton += - self.pulses[1](time) * qs
            hamilton += 1/2 * self.pulses[0](time) * xp___px
            hamilton += 1/2 * self.pulses[0](time) * pxp
            if nn_neighbors:
                hamilton += block / 2**power * q_q
            if nnn_neighbors:
                hamilton += block / 2**power * q__q
            if schrieffer_wolf:
                epsilon1 = -np.abs(self.pulses[0](time))**2 / (block - self.pulses[1](time)) / 4
                epsilon2 = -np.abs(self.pulses[0](time))**2 / (2 * block - self.pulses[1](time)) / 4

                hamilton += epsilon1 * sw1
                hamilton += epsilon2 * sw2
            return hamilton

        self.hamiltonian = hamiltonian_eff
        return self.hamiltonian


    def rydberg_3D_array_vdw_3l(self, block, block_cross, power=6):
        self.basis = np.eye(3, dtype=np.complex128)
        sigma_rg = np.outer(self.basis[1], self.basis[0])
        sigma_ar = np.outer(self.basis[2], self.basis[1])
        sigma_ga = np.outer(self.basis[0], self.basis[2])
        sigma_rr = np.outer(self.basis[1], self.basis[1])
        sigma_aa = np.outer(self.basis[2], self.basis[2])

        h_int = 0
        h_rg = h_gr = 0
        h_ar = h_ra = 0
        h_ga = h_ag = 0
        h_rr = h_aa = 0

        pairs = list(combinations(range(self.num_of_atoms), 2))
        for ii, jj in pairs:
            pos_i = self.positions[ii]
            pos_j = self.positions[jj]
            distance = np.linalg.norm(pos_i - pos_j)

            h_int += block / distance**power * self.__pair_expand__(matrix1=sigma_rr,
                                                                    matrix2=sigma_rr,
                                                                    from_subspaces=(ii, jj))
            h_int -= block / distance**power * self.__pair_expand__(matrix1=sigma_aa,
                                                                    matrix2=sigma_aa,
                                                                    from_subspaces=(ii, jj))
            h_int += block_cross / distance**power * self.__pair_expand__(matrix1=sigma_aa,
                                                                          matrix2=sigma_rr,
                                                                          from_subspaces=(ii, jj))
            h_int += block_cross / distance**power * self.__pair_expand__(matrix1=sigma_rr,
                                                                          matrix2=sigma_aa,
                                                                          from_subspaces=(ii, jj))

        for ii in range(self.num_of_atoms):
            h_rg += self.__expand__(sigma_rg,   from_subspace=ii)
            h_gr += self.__expand__(sigma_rg.T, from_subspace=ii)

            h_ar += self.__expand__(sigma_ar,   from_subspace=ii)
            h_ra += self.__expand__(sigma_ar.T, from_subspace=ii)

            h_ga += self.__expand__(sigma_ga,   from_subspace=ii)
            h_ag += self.__expand__(sigma_ga.T, from_subspace=ii)

            h_rr += self.__expand__(sigma_rr,   from_subspace=ii)
            h_aa += self.__expand__(sigma_aa,   from_subspace=ii)

        def hamiltonian_eff(time):
            hamilton = (1+0.j)*h_int

            hamilton += 1/2 * self.pulses[0](time) * h_rg
            hamilton += 1/2 * np.conj(self.pulses[0](time)) * h_gr

            hamilton += 1/2 * self.pulses[2](time) * h_ar
            hamilton += 1/2 * np.conj(self.pulses[2](time)) * h_ra

            hamilton += 1/2 * self.pulses[4](time) * h_ga
            hamilton += 1/2 * np.conj(self.pulses[4](time)) * h_ag

            hamilton -= self.pulses[1](time) * h_rr
            hamilton -= self.pulses[3](time) * h_aa
            hamilton -= self.pulses[5](time) * h_aa
            return hamilton

        self.hamiltonian = hamiltonian_eff
        return self.hamiltonian


    def rydberg_3D_array_vdw_motion(self, block, decay_r=0, power=6, vel=None, flip_b=False):
        self.basis = np.eye(2, dtype=np.complex128)
        sigma_rg = np.einsum('i,j->ij', self.basis[1], self.basis[0])
        sigma_rr = np.einsum('i,j->ij', self.basis[1], self.basis[1])

        h_rr  = 0
        h_rg  = h_gr = 0
        h_int = np.zeros((self.num_of_atoms, self.num_of_atoms, 2**self.num_of_atoms, 2**self.num_of_atoms),
                         dtype=np.complex128)

        pairs = list(combinations(range(self.num_of_atoms), 2))
        for ii, jj in pairs:
            h_int[ii, jj] = self.__pair_expand__(matrix1=sigma_rr,
                                                 matrix2=sigma_rr,
                                                 from_subspaces=(ii, jj))

        for ii in range(self.num_of_atoms):
            """∑ᵢΩ/2|r><1| + h.c - ∑ᵢ(Δ + iΓ/2)|r><r|"""
            h_rg += self.__expand__(sigma_rg, from_subspace=ii)
            h_gr += self.__expand__(sigma_rg.T, from_subspace=ii)
            h_rr += self.__expand__(sigma_rr, from_subspace=ii)

        def hamiltonian_eff(time):
            hamilton = 0

            _pairs = list(combinations(range(self.num_of_atoms), 2))
            for iii, jjj in _pairs:
                pos_i = self.positions[iii] + vel[iii] * time
                pos_j = self.positions[jjj] + vel[jjj] * time
                """∑∑ᵢⱼ Bᵢⱼ|rr><rr|    with     Bᵢⱼ = C₆/||xᵢ - xⱼ||⁶"""
                distance = np.linalg.norm(pos_i - pos_j)
                hamilton += block / distance**power * h_int[iii, jjj]

            sign_flip = +1
            if flip_b: sign_flip = -np.sign(time - self.times[-1] / 2)
            hamilton *= sign_flip

            hamilton += 1 / 2 * self.pulses[0](time) * h_rg
            hamilton += 1 / 2 * np.conj(self.pulses[0](time)) * h_gr
            hamilton -= (self.pulses[1](time) + 1.j * decay_r / 2) * h_rr
            return hamilton

        self.hamiltonian = hamiltonian_eff
        return self.hamiltonian



class QuantumSimulator(object):
    def __init__(self):
        self.history = []
        self.times   = []

    def f_propagate(self, input_state, hamiltonian, rtol=1e-9, atol=1e-9):
        """self.num_of_qubits = len(input_states.shape)"""
        self.times.sort()
        hold = input_state
        input_state = input_state.reshape(-1)

        def RHS(time, psi):
            psi = psi.reshape(hold.shape)
            return -1.j * hamiltonian(time) @ psi

        integrator = inte.RK45(fun    =RHS,
                               y0     =input_state,
                               t0     =self.times[0].item(),
                               t_bound=self.times[-1].item(),
                               rtol   =rtol,
                               atol   =atol,
                               vectorized=True)
        history = []
        times   = []
        history.append(integrator.y)
        times.append(integrator.t)
        while integrator.status != 'finished':
            integrator.step()
            history.append(integrator.y)
            times.append(integrator.t)
        self.history = np.array(history)
        self.times   = np.array(times)


    def b_propagate(self, target_state, hamiltonian, rtol=1e-9, atol=1e-9):
        self.times.sort()
        hold = target_state
        target_state = target_state.reshape(-1)

        def RHS(time, psi):
            psi = psi.reshape(hold.shape)
            return -1.j*hamiltonian(time)@psi
        integrator = inte.RK45(fun    =RHS,
                               y0     =target_state,
                               t0     =self.times[-1].item(),
                               t_bound=self.times[0].item(),
                               rtol   =rtol,
                               atol   =atol,
                               vectorized=True)
        history = []
        times   = []
        history.append(integrator.y)
        times.append(integrator.t)
        while integrator.status != 'finished':
            integrator.step()
            history.append(integrator.y)
            times.append(integrator.t)
        self.history = np.array(history)[::-1]
        self.times   = np.array(times)[::-1]

    def spectrum(self, hamiltonian):
        """Returns the instantaneous eigenvalues and eigenvectors of a time dependent
        hamiltonian, arranged from lowest energy to highest with the i-index:

        e_i(t)   = eig_val_list[t, i]
        v_i(t)_j = eig_vec_list[t,j,i]"""
        eig_val_list = []
        eig_vec_list = []

        for _time in self.times:
            eigenvalues, eigenvectors = np.linalg.eig(hamiltonian(_time))
            idx          = np.argsort(eigenvalues)
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            eig_val_list.append(eigenvalues)
            eig_vec_list.append(eigenvectors)
        return np.vstack(eig_val_list), np.array(eig_vec_list)


if __name__ == '__main__':
    pass