// Basic sync record script for RTL-SDR
// github.com/temataro/wfm-demod   2025
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <rtl-sdr.h>

#define DEFAULT_FC 106000000 // 106 MHz (Radio Two)
#define DEFAULT_SR 2400000 // 2.4 MSPS
#define DEFAULT_GAIN -10 // auto-gain

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

    /* --- Record to file */
    size_t buf_num_samples = 1 << 23;
    size_t BYTES_PER_SAMPLE = sizeof(int8_t) * 2; // I and then Q
    void *buf = calloc(buf_num_samples, BYTES_PER_SAMPLE);
    FILE *fp = fopen(outfile, "wb+");

    if (!fp)
    {
        printf("[FATAL] Error opening file! Terminating.\n");
    }

    int num_read = 0;
    r = rtlsdr_read_sync(dev, buf, buf_num_samples, &num_read);
    if (r == 0)
    {
        printf("Successfully read: %d samples from SDR.\n", num_read);
    }
    else
    {
        printf("Problem reading [ERR %d]! Goodbye.\n", r);
    }

    size_t nmemb_written = fwrite(buf, BYTES_PER_SAMPLE, buf_num_samples, fp);
    if (nmemb_written != buf_num_samples) {printf("[ERROR] Only wrote %zu/%zu samples from buf into file.\n", nmemb_written, buf_num_samples);}
    else {printf("Successfully wrote %zu samples to file %s.\n", nmemb_written, outfile);}
    /* --- */

    free(buf);
    fclose(fp);
    rtlsdr_close(dev);

    return EXIT_SUCCESS;
}

//    void *calloc(size_t nmemb, size_t size);

/* FILE *fopen(const char *restrict pathname, const char *restrict mode); */
// fopen()

/*
 size_t fwrite(const void ptr[restrict .size * .nmemb],
              size_t size, size_t nmemb,
              FILE *restrict stream);
*/
// fwrite()

/*
 * Read synchronously from RTL-SDR
 * rtlsdr_read_sync(
 *     rtlsdr_dev_t *dev,
 *     void *buf,
 *     int samples_to_read,
 *     int *num_samples_read_by_dev,
 * );
 */
