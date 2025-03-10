from math import pi
import numpy as np


def omega_square(times, start, duration, area=2 * pi):
    """
    Generate a square pulse with a specified area, duration, and start time.
    The pulse is normalized so that its integral over the duration equals the specified area.

    Parameters:
    -----------
    times : float or np.ndarray
        Time values at which to evaluate the pulse. Can be a single value or an array.
    start : float
        The start time of the pulse.
    duration : float
        The duration of the pulse.
    area : float, optional
        The total area of the pulse. Default is 2 * \(\pi\).

    Returns:
    --------
    np.ndarray
        The complex amplitude of the pulse at the specified times. The output has the same shape as `times`.
        Values are `area / duration` within the pulse duration and `0` elsewhere.

    """
    times = np.asarray(times)
    amplitude = area / duration
    pulse = np.where((times >= start) & (times < start + duration), amplitude, 0)
    return pulse.astype(complex)


def Omega_gaussian_pow(times, start, duration, dev, amplitude, power=4):
    """
    This method creates a single adiabatic pulse. The shape of the pulse
    is the shifted Gaussian curve with deviation = duration / 5, normalized
    to have a specific amplitude.

    Parameters:
    - times: float or ndarray, input time values
    - start: float, start time of the pulse
    - duration: float, duration of the pulse
    - dev: float, deviation for the Gaussian function
    - amplitude: float, the maximum amplitude of the pulse
    - power: int, power of the Gaussian term (default is 4)

    Returns:
    - Omega: complex128 or ndarray of complex128, the Gaussian pulse at the given time(s)
    """
    times = np.asarray(times)
    within_pulse = (start <= times) & (times <= start + duration)
    if np.any(within_pulse):
        nominator = np.exp(- ((times - start - duration / 2) ** power) / dev ** power) \
                    - np.exp(- (duration / 2) ** power / dev ** power)
        denominator = 1 - np.exp(- (duration / 2) ** power / dev ** power)
        result = np.where(within_pulse, amplitude * nominator / denominator, 0)
        return np.array(result, dtype=np.complex128)
    else:
        return np.array(0, dtype=np.complex128)


def ckz_gate_error(simulated, num_of_qubits=2, canonical=True):
    """
    Calculate the average error of a Controlled-k-Z (CkZ) quantum Gate by comparing a simulated
    unitary to the theoretical ideal Gate.

    Parameters:
    -----------
    simulated : np.ndarray
        A 2^N x 2^N complex matrix representing the simulated Gate, where N is the number of atoms.
    num_of_qubits : int, optional
        The number of qubits (atoms) involved in the Gate. Default is 2.
    canonical : bool, optional
        If True, use the canonical CkZ Gate with a -1 phase on the last state.
        If False, apply the phase on the first state and adjust signs accordingly. Default is True.

    Returns:
    --------
    float
        The error of the simulated Gate compared to the theoretical Gate, as 1 - fidelity.
    """
    hilbert_d = 2 ** num_of_qubits
    theoretical = np.eye(hilbert_d, dtype=np.complex128)
    if canonical:
        theoretical[-1, -1] = -1
    else:
        theoretical[0, 0] = -1
        theoretical *= -1
    m = np.dot(theoretical.conj().T, simulated)
    trace_m = np.trace(m)
    fidelity = (np.sum(np.abs(m)**2) + np.abs(trace_m)**2) / (hilbert_d * (hilbert_d + 1))
    return 1 - np.real(fidelity)
