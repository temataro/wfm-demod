
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <rtl-sdr.h>
#include <complex>
#include <array>

#define DEFAULT_FC 106000000 // 106 MHz (Radio Two)
#define DEFAULT_SR 2400000 // 2.4 MSPS
#define DEFAULT_GAIN -10 // auto-gain

const size_t buf_num_samples = 1 << 23;
typedef std::complex<float> cf32;

/* Prototype jail */
int read_buf_to_complex_arr(const int *buf, size_t buf_size,
                            std::array<cf32, buf_num_samples> &arr);
/* --- */

static void rtlsdr_callback(unsigned char *buf, uint32_t len, void *ctx)
{
    printf("Doing with one read!\n");
    if (ctx)
    {
        fwrite(buf, len, 1, (FILE *)ctx);
        printf("Done with one read  %d!\n", len);
    }
}

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
        std::cerr << "Failed to open RTL-SDR device #" << device_index << "\n";
        return EXIT_FAILURE;
    }
    /* --- */

    /* --- Configure device */
    rtlsdr_set_center_freq(dev, DEFAULT_FC);
    rtlsdr_set_sample_rate(dev, DEFAULT_SR);
    rtlsdr_set_tuner_gain_mode(dev, DEFAULT_GAIN == 0 ? 0 : 1);
    rtlsdr_set_tuner_gain(dev, DEFAULT_GAIN);

    std::cout << "Tuned to " << DEFAULT_FC / 1e6 << " MHz, "
              << DEFAULT_SR / 1e6 << " MS/s\n";
    /* --- */

    /* Reset endpoint before we start reading from it (mandatory) */
    rtlsdr_reset_buffer(dev);
    printf("Buffer reset successfully!\n");

    FILE *fp = fopen("out_async.iq", "wb+");
    // FILE *fopen(const char *restrict pathname, const char *restrict mode);
    int rc = rtlsdr_read_async(dev, rtlsdr_callback, fp, 0, 1 << 18);

    rtlsdr_close(dev);
    return rc;
}
