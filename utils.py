#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import numpy.typing as npt
import matplotlib.pyplot as plt


plt.style.use("./computermodern.mplstyle")

# constants
fs: float = 2e6  # samplerate, Hz
dt: float = 1 / fs

TAU: float = np.pi * 2


def gen_fm(
    f_delta: float = 0.2e6,
    f_c: float = 1e6,
    t_ramp: float = 10e-3,
    freq: npt.NDArray | None = None,
) -> npt.NDArray:
    """
    Frequency is the derivative of phase.

    for s(t) = A * cos((w_c + w_i(t))*t + phi)
             = A * cos(theta(t))

    integral of (w_c + w_i(t)) = theta(t)

    For a linear ramp, w_i(t) = u*t - w_init

                        u      - ramp rate
                        w_init - initial frequency

    theta(t) = w_c*t + (u/2)*t^2 - w_init*t + phi

    for a random signal, we can integrate using the scipy.integrate function.
    """

    if freq is None:
        N: int = int(fs * t_ramp)
        u: float = TAU * f_delta / t_ramp
        t: npt.NDArray = np.arange(0, N, dtype=np.float32) * dt
        phase: npt.NDArray = TAU * t + u * np.square(t) * 0.5 - TAU * f_delta * 0.5 * t

    else:
        phase = np.cumsum(freq)

    wrapped_phase = phase % TAU
    plt.plot(wrapped_phase)
    plt.show()
    plt.plot(np.cumsum(wrapped_phase))
    plt.show()

    sig: npt.NDArray = np.exp(-1j * phase)

    return sig


def main() -> None:
    t: npt.NDArray = np.arange(0, 100_000, dtype=np.float32) * dt

    def test_gen_fm():
        freq = 1 * np.sin(500 * t)
        sig = gen_fm(freq=freq)
        plt.specgram(sig, NFFT=512)
        plt.show()

    def test_integration():
        # just visual gutcheck that integration works
        sig = np.sin(100_000 *t)

        fig, axs = plt.subplots(nrows=2)

        axs[0].plot(sig)
        axs[1].plot(np.cumsum(sig))
        plt.show()

    # test_integration()
    test_gen_fm()


if __name__ == "__main__":
    main()
