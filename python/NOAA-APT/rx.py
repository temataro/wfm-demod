#!/usr/bin/env python3

"""
Demodulate WFM flavored APT signals from NOAA weather satellites
and __nothing__ more!
https://sourceforge.isae.fr/projects/weather-imgs-of-noaa-satellites/wiki/Apt_

NOAA signals are analog video transmission following the APT modulation scheme.
    - 2 channels of data: one for visible, one for IR.
    - BW 34KHz AM modulation with 256 levels for each pixel.
    - AM center frequency = 2.4kHz
    - 120 lines/minute -> 30 seconds/line
    - 2080 'bits' per line
        - 909 channel A img pixels           --- total = 909
        - 909 channel B img pixels           --- total = 1818
        - 47 'bits' for ch A space data markers -- total = 1865
        - 47 'bits' for ch B space data markers -- total = 1912
        - 39 'bits' for ch A sync               -- total = 1951
        - 39 'bits' for ch B sync               -- total = 1990
        - 45 'bits' for ch A telemetry         --- total = 2035
        - 45 'bits' for ch B telemetry         --- total = 2080

    2080 bits per 30 seconds => 69.3333 bits per second
    (30 * 11025) samples / 2080 bits = 159.0144 samples/bit

T = 1/4160 seconds
11,025/4160 = 2.65 samples = 1T

Plan of action.

- Obtain a full frame of IQ data (or wav for now).
    `wget https://project.markroland.com/weather-satellite-imaging/N18_4827.zip`
- View specgram here and in audacity
- Write a comment section about how demodulation works
- Implement demod and get to a bitstream waveform
- Threshold and convert waveform to bitstream
- Classify bitstream into sections

"""

import sys
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

from scipy.io import wavfile
from matplotlib.widgets import Slider

plt.style.use("./computermodern.mplstyle")

sr = 11.025e3  # Hz
LPF_ORDER = 9
f_3dB = 4_000  # Hz
BITS_PER_T = 11025 / 4160
SYNC_SEQ_LEN = 40

#  +---------------------------------------------------------------------+
#  | Use VERBOSITY to display logging info/plots to different yap levels |
#  |   0 -- production level, only errors, no debugging                  |
#  |   1 -- current debugs                                               |
#  |   2 -- basic status and warnings                                    |
#  |   3 -- np array shapes and whatnot                                  |
#  |   4 -- plots for intermediate results                               |
#  +---------------------------------------------------------------------+
if len(sys.argv) > 1:
    VERBOSITY = int(sys.argv[1])
else:
    VERBOSITY = 2


def demod_lpf(sig):
    # Demodulation would involve an envelope detector or a LPF
    b, a = sp.butter(
        N=4,
        Wn=[2400 / (2 * np.pi), 3600 / (2 * np.pi)],
        btype="bandpass",
        output="ba",
        fs=sr,
    )
    sig_lpf = sp.filtfilt(b, a, sig)
    # Normalize
    max_sig = np.max(sig_lpf)
    min_sig = np.min(sig_lpf)
    range_sig = max_sig - min_sig
    sig_lpf = (sig_lpf - min_sig) / range_sig

    w, h = sp.freqs(b, a)

    sig_section = sig[sig.size // 100 : sig.size // 100 + 5000]
    sig_lpf_section = sig_lpf[sig.size // 100 : sig.size // 100 + 5000]

    fig, axs = plt.subplots(nrows=2, figsize=(10, 6), dpi=160)
    ax_slider = plt.axes([0.25, 0.04, 0.65, 0.03])
    slider = Slider(ax_slider, "Frequency", 100, sr // 2 - 10, valinit=20)  # Freq in Hz

    def show_sig_spectrum():
        fig, ax = plt.subplots()
        ax.plot(
            np.linspace(-sr // 2, sr // 2, sig.size),
            20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig)))),
        )
        plt.show()

    def update(f_3dB):
        b, a = sp.butter(
            N=LPF_ORDER, Wn=[2400, 3600], btype="bandpass", output="ba", fs=sr
        )
        sig_lpf = sp.filtfilt(b, a, sig)

        sig_lpf_section = sig_lpf[sig.size // 100 : sig.size // 100 + 5000]
        w, h = sp.freqs(b, a)

        mag_resp = 20 * np.log10(abs(h))

        filt_sig_line.set_ydata(sig_lpf_section)
        filt_resp_line.set_ydata(mag_resp)

        axs[1].set_ylim([mag_resp - 10, mag_resp + 10])
        axs[1].axvline(2400)
        axs[1].axvline(3600)
        fig.canvas.draw_idle()

    (sig_line,) = axs[0].plot(sig_section, color="#0d0d0d", linestyle="dotted")
    (filt_sig_line,) = axs[0].plot(sig_lpf_section, color="firebrick", linestyle="-.")
    (filt_resp_line,) = axs[1].semilogx(
        w, 20 * np.log10(abs(h))
    )  # display filter mag response here
    axs[1].axvline(f_3dB)
    axs[1].grid(which="both", axis="both")

    slider.on_changed(update)
    plt.show()

    sync_A = make_APT_sync_A()
    # massage sig_lpf to be the same relative magnitude as sync_A

    # Correlate sync_A with sig_lpf

    corr = sp.correlate(sync_A, sig_lpf)
    plt.plot(corr)
    plt.show()


def make_APT_sync_A():
    """
    7 cycles of 1040 Hz square wave.
    39T long.
    start:
        4T off
        7 pulses (2T on then 2T off) = 28T
        8 T off
    off = 11/256 digitized level
    on = 244/256 digitized level
    """

    # Maybe work in 10xsr and decimate?
    sync_sig = np.zeros(int(SYNC_SEQ_LEN * 10 * BITS_PER_T) + 1, dtype=np.int16) + 11

    period_2t_10x = 53  # samples/2T times 10
    square_start = 2 * period_2t_10x  # also 4T times 10
    period_4t_10x = square_start
    for i in range(7):
        pulse_start = square_start + i * period_4t_10x
        pulse_stop = pulse_start + period_2t_10x
        sync_sig[pulse_start:pulse_stop] += 244

    decimated_sync_sig = sync_sig[5::10][
        : int(SYNC_SEQ_LEN * BITS_PER_T)
    ]  # resample_to_4160(sync_sig)
    if VERBOSITY == 3:
        plt.plot(sync_sig)
        plt.show()
        print(decimated_sync_sig.size)
        plt.plot(decimated_sync_sig)
        plt.show()

    # decimate the sync to 4.160 KHz like the main sig before returning
    # decimated_sync_sig = np.array([0, 0, 255, 255, 0, 0, 255, 255,
    #                  0, 0, 255, 255, 0, 0, 255, 255,
    #                  0, 0, 255, 255, 0, 0, 255, 255,
    #                  0, 0, 255, 255, 0, 0, 0, 0, 0,
    #                  0, 0, 0]) - 128
    return decimated_sync_sig


def resample_to_4160(sig):
    # Resample to 4160 Sps because then each sample corresponds to one 'word'
    sig_duration = sig.size / 11_025
    new_sig_size = int(sig_duration * 4_160)
    sig = sp.resample(sig, new_sig_size)

    return sig


def demod_abs(sig):
    # N = sig.size
    # t = np.arange(N) * 1/4160
    sig = resample_to_4160(sig)
    digitized_sig = np.abs(sig)
    digitized_sig = digitized_sig.astype(np.float64)
    digitized_sig /= np.max(digitized_sig)

    if VERBOSITY == 3:
        plt.plot(255 * digitized_sig)
        plt.show()

    return (255 * digitized_sig).astype(np.uint8)


def demod_hilbert(sig):
    """
    Procedure:
    Do a Hilbert transform and get IQ data out of the WFM demodulated signal.
    (Set the -ve frequencies to zero in the real FFT of the signal and iFFT.)
    """

    SIG = np.fft.fft(sig)
    SIG[: SIG.size // 2] = 0.0

    sig_iq = np.fft.ifft(SIG)

    def swap_iq(sig):
        # hypothesis: Swapping IQ pairs would give a nicer img (according to
        # other implementations I've seen online.)
        # (a + jb)* = (a - jb)
        # (a - jb) * j = (b + ja)
        return j * sig_iq.conj()

    sig_demod = np.abs(sig_iq)

    # Normalize to 256
    # sig_demod /= np.max(sig_demod)
    # sig_demod *= 256
    low, high = np.percentile(sig_demod, (5, 95))
    norm_demod = np.round((255 * (sig_demod - low)) / (high - low)).clip(0, 255)

    digitized_sig = norm_demod.astype(np.uint8)

    if VERBOSITY == 1:
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(sig_iq[:100].real, "-")
        axs[0].plot(sig_iq[:100].imag, "--")
        axs[1].plot(sig_demod[:10000], "-.")
        plt.show()

        print(digitized_sig[:1000])
        plt.stem(digitized_sig[:1000])
        plt.show()

    digitized_sig = resample_to_4160(digitized_sig)
    return digitized_sig


def hist_norm(img):
    """
    https://en.wikipedia.org/wiki/Histogram_equalization
    """

    orig_shape = img.shape
    img = img.flatten()
    N = img.size

    hist, _ = np.histogram(img, bins=256)
    pdf = hist / N

    def cdf(i):
        return np.sum(pdf[:i])

    cdf_x = [cdf(i) for i in range(256)]
    LUT = list(map(round, cdf_x * 255))
    # when you give an input value, you get the value that corresponds
    # to a normalized histogram mapping of the original img.

    # Here's a comparison between the un-equalized and equalized imgs.
    img_eq = np.array([LUT[int(img[i])] for i in range(N)])
    pdf_eq = np.histogram(img_eq)[0] / N
    cdf_y = [np.sum(pdf_eq[:i]) for i in range(256)]

    if VERBOSITY == 3:
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(pdf, "--")
        axs[0].plot(cdf_x, "-")
        axs[0].set_title("Original img PDF & CDF")
        axs[1].plot(pdf_eq, "--")
        axs[1].plot(cdf_y, "-")
        axs[1].set_title("Equalized img PDF & CDF")
        plt.show()

    return img_eq.reshape(orig_shape)


def align_with_sync(img, sync_sig):
    """
    Take a picture and realign so the sync signals come at the beginning column.
    """


    def moving_avg_filt(arr, order):
        kernel = np.ones(order) / order
        return np.convolve(arr, kernel, mode="valid")

    corr = np.zeros_like(img)
    print("CORR", corr.shape)
    print("SYNC", sync_sig.shape)
    print("IMG", img.shape)
    for r, row in enumerate(img):
        corr[r] = np.abs(
            np.convolve(row, sync_sig, mode="same")
        )  # This will be as big as row.size

    # This should show us how horizontally distorted the img is
    # plt.imshow(corr)
    # plt.show()

    MAF_ORDER = 2
    # only show corr peak in each row
    rolls = np.zeros(corr.shape[0])
    for r, row in enumerate(corr):
        idx = np.argmax(row)
        val = row[idx]
        corr[r] = np.zeros(row.size)
        corr[r][idx] = val

        if idx <= 1040:
            rolls[r] = 1040 - idx
        else:
            rolls[r] = 2080 - idx

    # plt.imshow(corr, cmap="turbo", interpolation="none")
    # plt.show()

    if VERBOSITY == 4:
        # plt.plot(sync_sig)
        # plt.show()

        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(img, cmap="Greys", interpolation="none")

        # possibly try a moving avg filter on roll indices before...
        rolls = moving_avg_filt(rolls, MAF_ORDER)
        for r, roll in enumerate(rolls):
            if abs(roll) < MAF_ORDER:
                continue
            img[r] = np.roll(img[r], roll)
        axs[1].imshow(img, cmap="Greys", interpolation="none")

        axs[0].set_title("img before sync alignment")
        axs[1].set_title("img after sync alignment")
        plt.show()

    return rolls


def main():
    samp, sig = wavfile.read("./N18_4827.wav", sr)

    assert samp == sr, "Sample rate not expected!"

    # words = demod_hilbert(sig)
    sig = sig.astype(np.float64)
    words = demod_abs(sig)
    # *** img pre-histogram normalization! ***
    num_lines = words.size // 2080 + 1
    img = np.zeros(num_lines * 2080)
    img[: words.size] = words.flatten()
    img = img.reshape((num_lines, 2080))
    print(img.shape)

    if VERBOSITY == 3:
        plt.imshow(img, cmap="Greys", interpolation="none")
        plt.show()

    sync_A = make_APT_sync_A()
    rolls = align_with_sync(img, sync_A)

    for r, roll in enumerate(rolls):
        img[r] = np.roll(img[r], roll)
    # Histogram normalization
    # Don't normalize both imgs at the same time!
    visible_img = img[:, :1040]
    ir_img = img[:, 1040:]
    visible_eq_img = hist_norm(visible_img)
    ir_eq_img = hist_norm(ir_img)
    eq_img = np.zeros_like(img)
    eq_img[:, :1040] = visible_eq_img
    eq_img[:, 1040:] = ir_eq_img

    if VERBOSITY == 3:
        plt.imshow(eq_img, cmap="Greys", interpolation="none")
        plt.show()
    # ---

if __name__ == "__main__":
    main()
