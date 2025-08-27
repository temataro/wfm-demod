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
 * What sort of analysis are we doing? (TODO)
 * 1. Converting to cf32 and doing FM demodulation through a phase
 * discriminator.
 * (Possibly take a half step to decimate/average to increase SNR)
 * 2. For NOAA-APT, do a AM demodulation followed by the Hilbert Method of APT
 * decoding.
 * 3. Feed a Linux audio device with a constant stream of FM data.
 */

// More concrete TODOs
// Decimate function to increase SNR
// Integrate ggplot graphing IQ data/Constellation diagrams

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>
#include <rtl-sdr.h>
#include <time.h>

#define DEFAULT_FC 106000000 // 106 MHz (Radio Two)
#define DEFAULT_SR 2400000 // 2.4 MSPS
#define DEFAULT_GAIN 0 // auto-gain
#define READ_SIZE 0x01 << 18 // 262,144 samples

typedef std::complex<float> cf32;

typedef struct
{
    uint64_t sample_counter;
    clock_t init_time;
    FILE *fp;
    FILE *benchmark_fp;
} sdr_ctx_t;

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx);
void read_to_vec(unsigned char *buf, uint32_t len, std::vector<cf32> &iq);

int main(int argc, char **argv)
{
    (void)argc;
    rtlsdr_dev_t *dev = nullptr;
    int device_index = 0;
    int r;
    char *outfile = argv[1];

    if (strcmp(outfile, "") == 0)
    {
        printf(
            "[FATAL] Input file not provided. Defaulting output to out.iq\n");
    }

    /* --- Check device for lice and ticks */
    char manufact[256] = {0};
    char product[256] = {0};
    char serial[256] = {0};
    r = rtlsdr_get_device_usb_strings(0, manufact, product, serial);

    printf("\n\n====\n[INFO] Device details: %d, \n"
           "Manufacturer: %s,\n"
           "Product: %s,\n"
           "Serial: %s\n====\n",
           r, manufact, product, serial);

    r = rtlsdr_open(&dev, device_index);
    if (r < 0)
    {
        fprintf(stderr, "[FATAL] Failed to open RTL-SDR device #%d\n",
                device_index);
        return EXIT_FAILURE;
    }
    /* --- */

    /* --- Configure device */
    rtlsdr_set_center_freq(dev, DEFAULT_FC);
    rtlsdr_set_sample_rate(dev, DEFAULT_SR);
    rtlsdr_set_tuner_gain_mode(dev, DEFAULT_GAIN == 0 ? 0 : 1);
    rtlsdr_set_tuner_gain(dev, DEFAULT_GAIN);

    printf("[INFO] Tuned to %.2f MHz. Sample rate: %.2f MSps.\n",
           DEFAULT_FC / 1e6, DEFAULT_SR / 1e6);
    /* --- */

    /* Reset endpoint before we start reading from it (mandatory) */
    rtlsdr_reset_buffer(dev);
    printf("Buffer reset successfully!\n");

    /* Populate sdr_ctx_t */
    sdr_ctx_t sdr_ctx;
    FILE *fp = fopen("test.iq", "wb+");
    if (!fp)
    {
        printf("[FATAL] Error opening file! Terminating.\n");
    }
    sdr_ctx.fp = fp;
    sdr_ctx.sample_counter = 0;
    sdr_ctx.init_time = clock();

    FILE *benchmark_fp = fopen("cb_bench.txt", "w+");
    if (!benchmark_fp)
    {
        printf("[FATAL] Error opening file! Terminating.\n");
    }
    sdr_ctx.benchmark_fp = benchmark_fp;
    /* --- */

    /*!
     * Read samples from the device asynchronously. This function will block
     *until it is being canceled using rtlsdr_cancel_async()
     *
     * \param dev the device handle given by rtlsdr_open()
     * \param cb callback function to return received samples
     * \param ctx user specific context to pass via the callback function
     * \param buf_num optional buffer count, buf_num * buf_len = overall buffer
     *                              size set to 0 for default buffer count (15)
     * \param buf_len optional buffer length, must be multiple of 512, should
     * be a multiple of 16384 (URB size), set to 0 for default buffer length
     * (16 * 32 * 512)
     * \return 0 on success
     */
    // RTLSDR_API int rtlsdr_read_async(rtlsdr_dev_t * dev,
    //                                  rtlsdr_read_async_cb_t cb, void *ctx,
    //                                  uint32_t buf_num, uint32_t buf_len);

    rtlsdr_read_async(dev, rtl_cb, &sdr_ctx,
                      0, // buf_num
                      READ_SIZE // buf_len
    );

    rtlsdr_close(dev);
    fclose(sdr_ctx.fp);
    fclose(sdr_ctx.benchmark_fp);

    return EXIT_SUCCESS;
}

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx)
{
    sdr_ctx_t *sdr_ctx = (sdr_ctx_t *)ctx;

    std::vector<cf32> iq((int)len / 2);
    read_to_vec(buf, len, iq);
    for (auto &z : iq)
    {
        printf("(%2.3f, %2.3f)\n", z.real(), z.imag());
    }

    size_t samp_written = fwrite(buf, 1, len, sdr_ctx->fp);
    // size_t fwrite(const void ptr[restrict .size * .nmemb],
    //          size_t size, size_t nmemb,
    //          FILE *restrict stream);

    if (samp_written < len)
    {
        fprintf(stderr,
                "[ERROR] Expected to write %u samples but only wrote %zu!\n",
                len, samp_written);
    }
    sdr_ctx->sample_counter += samp_written;

    clock_t time = clock();
    double proc_time = (time - sdr_ctx->init_time) * 1e9 / CLOCKS_PER_SEC;
    sdr_ctx->init_time = time;

    fprintf(sdr_ctx->benchmark_fp, "%.5f\n", proc_time);
    fflush(sdr_ctx->benchmark_fp); // Otherwise we won't see anything printed
                                   // when we Ctrl-C out of this program on
                                   // termination.

    fprintf(stderr, "[STATUS] Took %.5f ns to process. Wrote %lu samples.\r",
            proc_time, sdr_ctx->sample_counter);
}

/* DSP Functions */
void read_to_vec(unsigned char *buf, uint32_t len, std::vector<cf32> &iq)
{
    // Take a buffer and convert it to a std::complex vector
    float i_mean = 0;
    float q_mean = 0;
    for (size_t i = 0; i < len / 2; i++)
    {
        cf32 z{(float)buf[2 * i], (float)buf[2 * i + 1]};
        i_mean += z.real();
        q_mean += z.imag();
        iq[i] = z;
    }

    i_mean /= (len / 2);
    q_mean /= (len / 2);

    // Also do dc offset removal
    cf32 mean{i_mean, q_mean};
    std::transform(iq.begin(), iq.end(), iq.begin(),
                   [mean](auto z) { return z - mean; });
    /* --- */
}
