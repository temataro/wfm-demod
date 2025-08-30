#!/usr/bin/env python3

import sys
import numpy as np
import numpy.typing as npt
import scipy as sp
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

filename = sys.argv[1]

# Read as float32 little-endian (default on x86)
angle_diff = np.fromfile(filename, dtype=np.float32)

# LPF angle_diff
filt_ord: int = 3
fs: float = 2.4e6 / 6 # sample rate at which IQ signal was recorded
b, a = sp.signal.butter(
    N=filt_ord, Wn=80e3, btype="low", analog=False, output="ba", fs=fs
)
angle_diff_lpf = sp.signal.filtfilt(b, a, angle_diff)

plt.plot(angle_diff[:10_000], 'k-', alpha=0.7)
plt.plot(angle_diff_lpf[:10_000], 'r-.', alpha=0.7)
plt.show()


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

save_to_wav(angle_diff_lpf, "angle_diff.wav")
