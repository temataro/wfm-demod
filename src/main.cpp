// Basic sync record script for RTL-SDR
// github.com/temataro/wfm-demod   2025
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <rtl-sdr.h>
#include <time.h>

#define DEFAULT_FC 106000000 // 106 MHz (Radio Two)
#define DEFAULT_SR 2400000 // 2.4 MSPS
#define DEFAULT_GAIN 0 // auto-gain

typedef struct
{
    uint64_t sample_counter;
    clock_t init_time;
    FILE *fp;
} sdr_ctx_t;

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx);

int main(int argc, char **argv)
{
    (void)argc;
    rtlsdr_dev_t *dev = nullptr;
    int device_index = 0;
    int r;
    char *outfile = argv[1];

    if (strcmp(outfile, "") == 0)
    {
        printf("Input file not provided. Defaulting output to out.iq\n");
    }

    /* --- Check device for lice and ticks */
    char manufact[256] = {0};
    char product[256] = {0};
    char serial[256] = {0};
    r = rtlsdr_get_device_usb_strings(0, manufact, product, serial);

    printf("\n\n====\nDevice details: %d, \n"
           "Manufacturer: %s,\n"
           "Product: %s,\n"
           "Serial: %s\n====\n",
           r, manufact, product, serial);

    r = rtlsdr_open(&dev, device_index);
    if (r < 0)
    {
        fprintf(stderr, "Failed to open RTL-SDR device #%d\n", device_index);
        return EXIT_FAILURE;
    }
    /* --- */

    /* --- Configure device */
    rtlsdr_set_center_freq(dev, DEFAULT_FC);
    rtlsdr_set_sample_rate(dev, DEFAULT_SR);
    rtlsdr_set_tuner_gain_mode(dev, DEFAULT_GAIN == 0 ? 0 : 1);
    rtlsdr_set_tuner_gain(dev, DEFAULT_GAIN);

    printf("Tuned to %.2f MHz. Sample rate: %.2f MSps.\n", DEFAULT_FC / 1e6,
           DEFAULT_SR / 1e6);
    /* --- */

    /* Reset endpoint before we start reading from it (mandatory) */
    rtlsdr_reset_buffer(dev);
    printf("Buffer reset successfully!\n");

    /*!
     * Read samples from the device asynchronously. This function will block
     *until it is being canceled using rtlsdr_cancel_async()
     *
     * \param dev the device handle given by rtlsdr_open()
     * \param cb callback function to return received samples
     * \param ctx user specific context to pass via the callback function
     * \param buf_num optional buffer count, buf_num * buf_len = overall buffer
     *size set to 0 for default buffer count (15) \param buf_len optional
     *buffer length, must be multiple of 512, should be a multiple of 16384
     *(URB size), set to 0 for default buffer length (16 * 32 * 512) \return 0
     *on success
     */
    // RTLSDR_API int rtlsdr_read_async(rtlsdr_dev_t * dev,
    //                                  rtlsdr_read_async_cb_t cb, void *ctx,
    //                                  uint32_t buf_num, uint32_t buf_len);

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
    /* --- */

    rtlsdr_read_async(dev, rtl_cb, &sdr_ctx, 0, 0x01 << 18);

    rtlsdr_close(dev);

    return EXIT_SUCCESS;
}

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx)
{
    sdr_ctx_t *sdr_ctx = (sdr_ctx_t *)ctx;
    size_t samp_written = fwrite(buf, 1, len, sdr_ctx->fp);
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

    fprintf(stderr, "[STATUS] Took %.3f ns to process. Wrote %lu samples.\r", proc_time, sdr_ctx->sample_counter);
    // size_t fwrite(const void ptr[restrict .size * .nmemb],
    //          size_t size, size_t nmemb,
    //          FILE *restrict stream);
}
