#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- CONFIG ---
FILENAME = "../xab"   # path to your float32 file
FS = 2.4e6               # sampling frequency (Hz) — change as appropriate
IS_COMPLEX = False       # set True if samples are interleaved complex (I/Q)
# ----------------

# Load samples
data = np.fromfile(FILENAME, dtype=np.float32)

if IS_COMPLEX:
    data = data.view(np.complex64)

# Compute Welch PSD
f, Pxx = welch(
    data,
    fs=FS,
    nperseg=2 ** 18,
    scaling='density',
    return_onesided=not IS_COMPLEX,
)

# Convert to dB
Pxx_dB = 10 * np.log10(Pxx + 1e-12)

# --- Plot ---
plt.style.use('../../computermodern.mplstyle')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(f, Pxx_dB, linewidth=1.0, color='k')

ax.set_title("Welch Power Spectral Density", fontsize=14)
ax.set_xlabel("Frequency [Hz]", fontsize=12)
ax.set_ylabel("Power Spectral Density [dB/Hz]", fontsize=12)
ax.set_xlim([f[10], 70e3])
ax.set_ylim([-75, -55])
ax.axvspan(19e3,55e3, color='r', alpha=0.1)
ax.axvspan(55e3,59e3, color='b', alpha=0.1)
ax.axvline(19e3, ls='dashdot', alpha=0.4, color='r', lw=2, label="19 KHz pilot tone (Mono)")
ax.axvline(38e3, ls='dashdot', alpha=0.4, color='b', lw=2, label="38 KHz pilot tone (Stereo)")
ax.axvline(57e3, ls='dashdot', alpha=0.4, color='k', lw=2, label="RDS signal")
ax.axvline(67e3, ls='dashdot', alpha=0.4, color='g', lw=2, label="DirectBand")
ax.grid(True, which='both', ls=':', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

