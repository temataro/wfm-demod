#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.integrate
import numpy.typing as npt
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

plt.style.use("./computermodern.mplstyle")

DBG_PLT: bool = False
# constants
fs: float = 1.92e6  # samplerate, Hz
dt: float = 1 / fs
wfm_bandwidth: float = 100e3

TAU: float = np.pi * 2


def read_iq(
    filename: str, samples_to_read: int | float = -1 / 2, offset=0, fs=fs
) -> npt.NDArray:
    """
    Read data from a recorded sc16 or cf32 binary file.
    Accepts offsets (in bytes) to see where to start the file from.
    Next offset should be count * bytes per count after the current read.
    """

    with open(filename, "rb") as file:
        samples_IQ: int = int(2 * samples_to_read)

        if ".sc16" in filename or ".cs16" in filename:
            sig: npt.NDArray = np.fromfile(
                file, dtype=np.int16, count=samples_IQ, offset=offset
            )
            ftype: str = "sc16"
            sig = sig.reshape(sig.size // 2, 2)
            # Convert to floats
            sig = sig.astype(np.float32)

            # remove dc offset
            sig[:, 0] -= np.mean(sig[:, 0])
            sig[:, 1] -= np.mean(sig[:, 1])
            sig = sig[:, 0] + 1j * sig[:, 1]

        else:
            sig = np.fromfile(file, dtype=np.complex64, count=samples_IQ, offset=offset)
            # remove dc offset
            sig.real -= np.mean(sig.real)
            sig.imag -= np.mean(sig.imag)
            ftype = "cf32"


        sig /= np.linalg.norm(sig)

    print(
        f"[STATUS]\
    {ftype} file of size {sig.size} read from binary {filename}.\
    ({sig.size / fs:.2f} seconds at {fs/1e6:.2f} MHz samp rate.)"
    )

    return sig


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

    if DBG_PLT:
        plt.plot(phase)
        plt.show()
        plt.plot(np.cumsum(phase))
        plt.show()

    sig: npt.NDArray = np.exp(-1j * phase)

    return sig


def demod_fm(sig: npt.NDArray) -> npt.NDArray:
    angle_diff: npt.NDArray = np.diff(np.angle(sig))
    # wrap angle_diff
    angle_diff = np.where(angle_diff > TAU / 2, 0, angle_diff)
    angle_diff = np.where(angle_diff < -TAU / 2, 0, angle_diff)

    # LPF angle_diff
    filt_ord: int = 11
    b, a = sp.signal.butter(
        N=filt_ord, Wn=200e3, btype="low", analog=False, output="ba", fs=fs
    )
    angle_diff_lpf = sp.signal.filtfilt(b, a, angle_diff)

    if DBG_PLT:
        fig, axs = plt.subplots(nrows=4)
        axs[0].plot(sig.real)
        axs[1].plot(np.angle(sig))
        axs[2].plot(angle_diff)
        axs[3].plot(angle_diff_lpf)
        plt.show()

    return angle_diff_lpf


<<<<<<< HEAD
def view_filter(b: npt.NDArray, a: npt.NDArray) -> None:
    w, h = sp.signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title("Butterworth filter frequency response")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Amplitude [dB]")
    plt.margins(0, 0.1)
    plt.grid(which="both", axis="both")
    plt.axvline(100, color="green")  # cutoff frequency
    plt.show()


def save_to_wav(audio: npt.NDArray, outfile: str) -> None:
    # Ensure that highest value is in 16-bit range
    demod_fs: int = int(fs)
    audio = audio * (2**15 - 1) / np.max(np.abs(audio))
    audio = audio.astype(np.int16)

    write(outfile, demod_fs, audio)


def center_transmission(sig: npt.NDArray, fc: float) -> npt.NDArray:
    """Center and filter a signal so the transmission desired shows up at 0 Hz offset."""

    t: npt.NDArray = np.arange(0, sig.size, dtype=np.float32) * dt
    sig *= np.exp(1j*TAU*fc*t)
    filt_ord: int = 5
    b, a = sp.signal.butter(
        N=filt_ord, Wn=wfm_bandwidth, btype="low", analog=False, output="ba", fs=fs
    )
    # view_filter(b, a)
    fm_filt = sp.signal.filtfilt(b, a, sig)

    if DBG_PLT:
        fig, axs = plt.subplots(nrows=2)
        axs[0].specgram(sig, cmap="turbo")
        axs[1].specgram(fm_filt, cmap="turbo")
        plt.show()

    return fm_filt

||||||| parent of 2d646791ff53 (Demoulate real FM IQ data offline)
=======
def view_filter(b: npt.NDArray, a: npt.NDArray) -> None:
    w, h = sp.signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()


def save_to_wav(audio: npt.NDArray) -> None:
    # Ensure that highest value is in 16-bit range
    demod_fs: int = int(fs)  # or maybe 48 KHz, try both
    audio = audio * (2**15 - 1) / np.max(np.abs(audio))
    audio = audio.astype(np.int16)

    write("output.wav", demod_fs, audio)

>>>>>>> 2d646791ff53 (Demoulate real FM IQ data offline)
def main() -> None:
    t: npt.NDArray = np.arange(0, 100_000, dtype=np.float32) * dt
    freq = 1 * np.sin(500 * t)

    def test_gen_fm():
        sig = gen_fm(freq=freq)
        plt.specgram(sig, NFFT=512)
        plt.show()

    def test_integration():
        # just visual gutcheck that integration works
        sig = np.sin(100_000 * t)

        fig, axs = plt.subplots(nrows=2)

        axs[0].plot(sig)
        axs[1].plot(np.cumsum(sig))
        plt.show()

    def test_demod(sig):
        demod_fm(sig)

    fm: npt.NDArray = read_iq("./assets/analog_FM_France.sigmf-data")
    # plt.specgram(fm, cmap="turbo"); plt.show()

    filt_ord: int = 11
    b, a = sp.signal.butter(
        N=filt_ord, Wn=wfm_bandwidth, btype="low", analog=False, output="ba", fs=fs
    )
    # view_filter(b, a)
    fm_filt = sp.signal.filtfilt(b, a, fm)
    # plt.specgram(fm, cmap="turbo"); plt.show()
    # plt.specgram(fm_filt, cmap="turbo"); plt.show()

    sig = demod_fm(fm_filt)
    save_to_wav(sig)


if __name__ == "__main__":
    main()
