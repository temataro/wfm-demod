// Basic async record script for RTL-SDR
// github.com/temataro/wfm-demod   2025

/* Some back of the envelope calculations to drive the architecture of this
 * program:
 * At a maximum sample rate of 2.4 MSps, and with the async read capturing
 * 2 ^ 18 bytes into the buffer...
 * 2 bytes per sample -> 2 ^ 17 (=131,072) samples per call to rtl_cb()
 * Each collection would take 2 ^ 17 / 2.4 microseconds (~54,600 us)
 *
 * (Hypothesis)
 * This means we can keep the program realtime as long as the processing we
 * take on each buffer doesn't last longer than say 50,000 us.
 *
 */

// TODOs
// Decimate function to increase SNR
// Integrate ggplot graphing IQ data/Constellation diagrams

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <pulse/error.h>
#include <pulse/simple.h>
#include <ranges>
#include <rtl-sdr.h>
#include <time.h>
#include <vector>

#include "utils/constants.hpp"
#include "wfm.h"

void rtl_cb(unsigned char* buf, uint32_t len, void* ctx);
void* run_radio(void* args);
void* control_radio(void* args);

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
int pending_retune = 1; // The atomic we want to secure with the mutex lock
/*
 pthread_mutex_lock(&lock);
 // Operate on pending_retune inside.
 pthread_mutex_unlock(&lock);
*/

bool SAVE_TO_DISK = 0;
int main(int argc, char** argv) {
    (void)argc;

    radio_params_t radio_params;
    radio_params.fc = DEFAULT_FC;
    radio_params.fs = DEFAULT_SR;
    radio_params.g = DEFAULT_GAIN;
    radio_params.argvs = (void*)argv;

    pthread_t radio_thread;
    pthread_t control_thread;

    pthread_create(&radio_thread, NULL, run_radio, (void*)&radio_params);
    pthread_create(&control_thread, NULL, control_radio, (void*)&radio_params);

    pthread_join(radio_thread, NULL);
    pthread_join(control_thread, NULL);

    return 0;
}

void* control_radio(void* args) {
    //
    radio_params_t* radio_params = (radio_params_t*)(args);
    int i=0;
    for (;;) {
        // Just retune every 4 seconds for now
        sleep(4);
        pthread_mutex_lock(&lock);
        pending_retune = i % 2;
        uint64_t fc = rand() % 2 ? 100'500'000 : 106'000'000;

        fprintf(stderr, "Attempting to retune to %lu\n", fc);
        fflush(stderr);
        radio_params->fc = fc;
        pthread_mutex_unlock(&lock);
        i++;
    }
}

void* run_radio(void* args) {
    radio_params_t* radio_params = (radio_params_t*)(args);
    char** argvs = (char**)(radio_params->argvs);
    char* outfile = argvs[1];

    if (strcmp(outfile, "") == 0) {
        ERR_PRINT("Input file not provided. Defaulting output to out.iq\n");
    }

    rtlsdr_dev_t* dev = nullptr;
    /* --- Check device for lice and ticks */
    int device_index = 0;
    int r;
    char manufact[256] = {0};
    char product[256] = {0};
    char serial[256] = {0};
    r = rtlsdr_get_device_usb_strings(0, manufact, product, serial);

    fprintf(stderr,
            GRN "====\nDevice details: %d,\n===\n" RST "Manufacturer: %s,\n"
                "Product: %s,\n"
                "Serial: %s\n====\n\n",
            r, manufact, product, serial);

    r = rtlsdr_open(&dev, device_index);
    radio_params->dev = dev;
    if (r < 0) {
        ERR_PRINT("Failed to open RTL-SDR device #%d", device_index);
        return nullptr;
    }

    /* --- */

    /* --- Configure device */
    rtlsdr_set_center_freq(dev, radio_params->fc);
    rtlsdr_set_sample_rate(dev, radio_params->fs);
    rtlsdr_set_tuner_gain_mode(dev, radio_params->g == 0 ? 0 : 1);
    rtlsdr_set_tuner_gain(dev, radio_params->g);

    INFO_PRINT("Tuned to %.2f MHz. Sample rate: %.2f MSps.", DEFAULT_FC / 1e6,
               DEFAULT_SR / 1e6);

    /* Reset endpoint before we start reading from it (mandatory) */
    rtlsdr_reset_buffer(dev);
    INFO_PRINT("Buffer reset successfully!");
    /* --- */

    /* Populate sdr_ctx_t */
    sdr_ctx_t sdr_ctx;
    FILE* fp = fopen("test.iq", "wb+");
    if (!fp) {
        ERR_PRINT("Error opening file! Terminating.");
    }
    sdr_ctx.fp = fp;
    sdr_ctx.sample_counter = 0;
    sdr_ctx.init_time = clock();

    FILE* benchmark_fp = fopen("cb_bench.txt", "w+");
    if (!benchmark_fp) {
        ERR_PRINT("Error opening file! Terminating.");
    }
    sdr_ctx.benchmark_fp = benchmark_fp;
    sdr_ctx.device = dev;

    // pulse audio device
    static const pa_sample_spec ss = {.format = PA_SAMPLE_S16LE,
                                      .rate = AUDIO_SR,
                                      .channels = NUM_AUDIO_CHAN};
    pa_simple* s = pa_simple_new(NULL,            // Use default server
                                 "WFM Demod App", // Application Name
                                 PA_STREAM_PLAYBACK,
                                 NULL,        // Use default device
                                 "La Musica", // Description of stream
                                 &ss,         // Sample format
                                 NULL, NULL,
                                 NULL // ignore error code
    );
    sdr_ctx.s = s;

    for (;;) {
        if (pending_retune) {
            rtlsdr_set_center_freq(dev, radio_params->fc);
            rtlsdr_set_sample_rate(dev, radio_params->fs);
            rtlsdr_set_tuner_gain_mode(dev, radio_params->g == 0 ? 0 : 1);
            rtlsdr_set_tuner_gain(dev, radio_params->g);

            printf("Retuning to fc=%ld\n", radio_params->fc);
            fflush(stdout);

            pthread_mutex_lock(&lock);
            pending_retune = 0;
            pthread_mutex_unlock(&lock);
        } else {
            printf("Staying at fc=%ld\n", radio_params->fc);
            fflush(stdout);
            rtlsdr_read_async(dev, rtl_cb, &sdr_ctx,
                              0,        // buf_num
                              READ_SIZE // buf_len
            );
        }
    }

    pa_simple_drain(s, NULL);
    pa_simple_free(s);

    rtlsdr_close(dev);
    fclose(sdr_ctx.fp);
    fclose(sdr_ctx.benchmark_fp);

    return nullptr;
}

void rtl_cb(unsigned char* buf, uint32_t len, void* ctx) {
    sdr_ctx_t* sdr_ctx = (sdr_ctx_t*)ctx;

    // DSP
    std::vector<cf32> iq((int)len / 2);
    read_to_vec(buf, len, iq);
    // ARR_PRINT(iq);

    std::vector<float> angle_diff = phase_diff_wrapped(iq);

    /* Note 2 */
    const size_t len_audio_buffer = 2622; // 2622 * 50 = 131,100 -- roughly
                                          // how many samples we get from the
                                          // SDR callback at once...
    /*
     * Apply polyphase resampling or any other DSP to lowpass/process the audio
     * signal.
     */

    std::vector<float> angle_diff_lpf =
        gptconvolve(angle_diff, filters::human_lpf);

    int16_t audio_buffer[len_audio_buffer];
    size_t samples_to_decimate = angle_diff_lpf.size();

    float val;
    size_t i = 0;
    while ((i < samples_to_decimate) && (i < len_audio_buffer)) {
        val = angle_diff_lpf[i * decimation_value];
        val /= PI;
        val *= (AUDIO_VOLUME * FULL_SCALE_AUDIO);
        audio_buffer[i] = (int16_t)val;

        i += 1;
    }

    /* Note 1 */

    int error;
    if (pa_simple_write(sdr_ctx->s, audio_buffer,
                        len_audio_buffer * NUM_AUDIO_CHAN * sizeof(int16_t),
                        &error) < 0) {
        ERR_PRINT("pa_simple_write: %s\n", pa_strerror(error));
    }
    /* *** --- *** */

#if SAVE_TO_DISK
    INFO_PRINT("Saving to disk.\n");
    size_t samp_written = 0;
    fwrite(buf, 1, len, sdr_ctx->fp);
    save_interleaved_cf32(iq, "out.cf32");
    save_floats(angle_diff, "angle_diffs.f32");
    save_floats(angle_diff_lpf, "angle_diff_lpf.f32");

    if (samp_written < len) {
        ERR_PRINT("Expected to write %u samples but only wrote %zu!", len,
                  samp_written);
    }
    sdr_ctx->sample_counter += samp_written;

    // END OF CB  -- Cleanup and profiling
    clock_t time = clock();
    double proc_time = (time - sdr_ctx->init_time) * 1e9 / CLOCKS_PER_SEC;
    sdr_ctx->init_time = time;

    fprintf(sdr_ctx->benchmark_fp, "%.5f\n", proc_time);
    fflush(sdr_ctx->benchmark_fp); // Otherwise we won't see anything printed
                                   // when we Ctrl-C out of this program on
                                   // termination.

    fprintf(stderr, "[STATUS] Took %.5f ns to process. Wrote %lu samples.\r",
            proc_time, sdr_ctx->sample_counter);
#endif

    // After we start running this we check if there is a pending
    // retune requested from the control thread.
    if (pending_retune)
    {
        rtlsdr_cancel_async(sdr_ctx->device);
    }
}
/*
     NOTES:
    1. Goal is to eventually get to
    float audio_buffer_f[len_audio_buffer];
    angle_diff.resize(131100, 0.0f); // zero pad to get to % 50 == 0
    polyphase_resample(fir_taps, angle_diff, audio_buffer_f);

    to avoid wasting 49/50 of the samples we LPF'd
    for now just decimate after collecting.

    2. Play out through pulseaudio
    We have 2^18 samples that we'll chunk into 2^10 sized sections
    (Thereby giving us READ_SIZE / len_audio_buffer = 2^8=64 sections)
    For each section we'll index into angle_diff at a different start and
    end point so we get 1024 fresh sections in the buffer.

    We're recording at 2.4MSps, not playing back at the same rate
    For now we'll do a 'naive' decimation where I'll just take one every
    50 every samples. Later, we'll want to polyphase resample by the same
    decimation value.
 */
