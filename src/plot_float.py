#!/usr/bin/env python3

import sys
import numpy as np
import numpy.typing as npt
import scipy as sp
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

plt.style.use("ggplot")
filename = sys.argv[1]
fs = 2.4e6

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

        elif ".sc8" in filename or ".cs8" in filename:
            sig = np.fromfile(
                file, dtype=np.int8, count=samples_IQ, offset=offset
            )
            ftype = "sc8"

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
    \n({sig.size / fs:.2f} seconds at {fs/1e6:.2f} MHz samp rate.)"
    )

    return sig


# if '--dbg' in sys.argv:
#     plt.plot(angle_diff[:10_000], 'k-', alpha=0.7)
#     plt.plot(angle_diff_lpf[:10_000], 'r-.', alpha=0.7)
#     plt.show()


def save_to_wav(audio: npt.NDArray, outfile: str) -> None:
    # Ensure that highest value is in 16-bit range
    demod_fs: int = int(fs)
    target_fs: int = 44_100
    downsample_by: int = demod_fs // target_fs
    print(f"{downsample_by=}")
    audio = sp.signal.resample_poly(audio, up=1, down=downsample_by,window=("kaiser", 8.6))
    audio = audio * (2**15 - 1) / np.max(np.abs(audio))
    audio = audio.astype(np.int16)

    write(outfile, target_fs, audio)

# save_to_wav(angle_diff, "angle_diff-nolpf.wav")

def main():
    sig = read_iq(sys.argv[1])
    f, Pxx = sp.signal.welch(sig, fs=fs, window="hamming", return_onesided=True, nperseg=2**15, noverlap=2**15 // 8)
    peaks, _ = sp.signal.find_peaks(Pxx, threshold=np.max(Pxx) * 0.3)
    print(peaks)
    plt.semilogy(f, Pxx)
    for p, peak in enumerate(peaks):
        plt.axvline(x=f[peak], ymax=Pxx[peak], c="blue")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
