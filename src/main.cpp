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
#include <ranges>
#include <fstream>
#include <cassert>
#include <pulse/simple.h>
#include <pulse/error.h>
#include <rtl-sdr.h>
#include <time.h>

// clang-format off
#define DEFAULT_FC          106'000'000 // 106 MHz (Radio Two)
#define DEFAULT_SR          2'400'000   // 0.4 MSPS
#define DEFAULT_GAIN        0           // auto-gain
#define READ_SIZE           0x01 << 18  // 262,144 samples

#define FULL_SCALE_AUDIO    32768.f
#define NUM_AUDIO_CHAN      1
#define AUDIO_SR            48000
#define AUDIO_VOLUME        80.f/100.f

/* Convenience macros */
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define RESET   "\033[0m"

#define PI      3.1415926535898f
#define RAD2DEG 180 / PI
#define DEG2RAD PI / 180
#define APPEND_FLAG std::ios::app // For fstream file writing operations
// clang-format off

// Macros for colored fprintf
#define ERR_PRINT(fmt, ...) \
    fprintf(stderr, RED "[FATAL]" fmt RESET "\n", ##__VA_ARGS__)

#define INFO_PRINT(fmt, ...) \
    fprintf(stdout, GREEN "[INFO]" fmt RESET "\n", ##__VA_ARGS__)

std::vector<std::string> LINE_BREAKS = {"\t", "\t", "\t", "\t", "\t", "\n"};
size_t NUM_ELTS_PER_LINE = LINE_BREAKS.size();

#define ARR_PRINT(arr) \
    for (auto [e, elt] : std::views::enumerate(arr)) \
        printf(BLUE "(%+2.4f, %+2.4f)" "%s" RESET ,elt.real(), elt.imag(), LINE_BREAKS[e % NUM_ELTS_PER_LINE].c_str())

typedef std::complex<float> cf32;
/* --- */

/* Prototype jail */
typedef struct
{
    uint64_t sample_counter;
    clock_t init_time;
    FILE *fp;
    FILE *benchmark_fp;
    pa_simple *s;
} sdr_ctx_t;

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx);
void read_to_vec(unsigned char *buf, uint32_t len, std::vector<cf32> &iq);
void save_interleaved_cf32(const std::vector<cf32> &iq, const std::string &filename);
void save_floats(const std::vector<float> &sig, const std::string &filename);
std::vector<float> phase_diff_wrapped(const std::vector<cf32> &iq);
/* --- */

int main(int argc, char **argv)
{
    (void)argc;
    rtlsdr_dev_t *dev = nullptr;
    int device_index = 0;
    int r;
    char *outfile = argv[1];

    if (strcmp(outfile, "") == 0)
    {
        ERR_PRINT("Input file not provided. Defaulting output to out.iq");
    }

    /* --- Check device for lice and ticks */
    char manufact[256] = {0};
    char product[256] = {0};
    char serial[256] = {0};
    r = rtlsdr_get_device_usb_strings(0, manufact, product, serial);

    INFO_PRINT("\n\n====\nDevice details: %d, \n"
               "Manufacturer: %s,\n"
               "Product: %s,\n"
               "Serial: %s\n====\n",
               r, manufact, product, serial);

    r = rtlsdr_open(&dev, device_index);
    if (r < 0)
    {
        ERR_PRINT("Failed to open RTL-SDR device #%d", device_index);
        return EXIT_FAILURE;
    }
    /* --- */

    /* --- Configure device */
    rtlsdr_set_center_freq(dev, DEFAULT_FC);
    rtlsdr_set_sample_rate(dev, DEFAULT_SR);
    rtlsdr_set_tuner_gain_mode(dev, DEFAULT_GAIN == 0 ? 0 : 1);
    rtlsdr_set_tuner_gain(dev, DEFAULT_GAIN);

    INFO_PRINT("Tuned to %.2f MHz. Sample rate: %.2f MSps.", DEFAULT_FC / 1e6,
               DEFAULT_SR / 1e6);
    /* --- */

    /* Reset endpoint before we start reading from it (mandatory) */
    rtlsdr_reset_buffer(dev);
    INFO_PRINT("Buffer reset successfully!");

    /* Populate sdr_ctx_t */
    sdr_ctx_t sdr_ctx;
    FILE *fp = fopen("test.iq", "wb+");
    if (!fp)
    {
        ERR_PRINT("Error opening file! Terminating.");
    }
    sdr_ctx.fp = fp;
    sdr_ctx.sample_counter = 0;
    sdr_ctx.init_time = clock();

    FILE *benchmark_fp = fopen("cb_bench.txt", "w+");
    if (!benchmark_fp)
    {
        ERR_PRINT("Error opening file! Terminating.");
    }
    sdr_ctx.benchmark_fp = benchmark_fp;

    // pulse audio device
    static const pa_sample_spec ss = {
        .format = PA_SAMPLE_S16LE,
        .rate = AUDIO_SR,
        .channels = NUM_AUDIO_CHAN
    };
    pa_simple *s = pa_simple_new(
            NULL,               // Use default server
            "WFM Demod App",    // Application Name
            PA_STREAM_PLAYBACK,
            NULL,               // Use default device
            "La Musica",        // Description of stream
            &ss,                // Sample format
            NULL,
            NULL,
            NULL                // ignore error code
    );
    sdr_ctx.s = s;
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
    pa_simple_drain(s, NULL);
    pa_simple_free(s);

    rtlsdr_close(dev);
    fclose(sdr_ctx.fp);
    fclose(sdr_ctx.benchmark_fp);

    return EXIT_SUCCESS;
}

void rtl_cb(unsigned char *buf, uint32_t len, void *ctx)
{
    sdr_ctx_t *sdr_ctx = (sdr_ctx_t *)ctx;

    // DSP
    std::vector<cf32> iq((int)len / 2);
    read_to_vec(buf, len, iq);
    // ARR_PRINT(iq);

    size_t samp_written = fwrite(buf, 1, len, sdr_ctx->fp);

    std::vector<float> angle_diff = phase_diff_wrapped(iq);

    // Play out through pulseaudio
    // We have 2^18 samples that we'll chunk into 2^10 sized sections
    // (Thereby giving us READ_SIZE / len_audio_buffer = 2^8=64 sections)
    // For each section we'll index into angle_diff at a different start and
    // end point so we get 1024 fresh sections in the buffer.
    //
    // We're recording at 2.4MSps, not playing back at the same rate
    // For now we'll do a 'naive' decimation where I'll just take one every
    // 50 every samples. Later, we'll want to polyphase resample by the same
    // decimation value.
    size_t decimation_value = DEFAULT_SR / AUDIO_SR;

    size_t num_samples = angle_diff.size();
    const size_t len_audio_buffer = 0x01 << 11;
    int16_t audio_buffer[len_audio_buffer];

    int error;
    // TODO: just to test, let's also write this audio_buffer out to a file
    // and see if it sounds like anything...
    std::ofstream ofs("audio_buffer.pcm", std::ios::binary | APPEND_FLAG);
    for (size_t i = 0; i < num_samples /(decimation_value * len_audio_buffer); i++)
    {
        int idx_start = i * len_audio_buffer * decimation_value;
        for (size_t j = 0; j < len_audio_buffer; j++)
        {
            int16_t val = static_cast<int16_t> (angle_diff[j * decimation_value + idx_start] * (AUDIO_VOLUME * FULL_SCALE_AUDIO) / PI);
            audio_buffer[j] = val;
            ofs.write(reinterpret_cast<const char *> (&val), sizeof(int16_t));
        }
        if(
                pa_simple_write(
                sdr_ctx->s,
                audio_buffer,
                len_audio_buffer * NUM_AUDIO_CHAN * sizeof(int16_t),
                &error
                ) < 0)
        {
            ERR_PRINT("pa_simple_write: %s\n", pa_strerror(error));
        }
    }

    save_interleaved_cf32(iq, "out.cf32");
    save_floats(angle_diff, "angle_diffs.f32");
    if (samp_written < len)
    {
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
}

void save_interleaved_cf32(const std::vector<cf32> &iq, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary | APPEND_FLAG);

    for (auto& elt: iq)
    {
        float i = elt.real();
        float q = elt.imag();

        ofs.write(
                reinterpret_cast<const char *> (&i), sizeof(float)
                );
        ofs.write(
                reinterpret_cast<const char *> (&q), sizeof(float)
                );
 /*     basic_ostream& write( const char_type* s, std::streamsize count );
  *     s: char pointer to the first byte of the variable's address
  *     count: the number of bytes to copy over starting from the address pointed by s
  *
  *    outputs the characters from successive locations in the character array
  *    whose first element is pointed to by s. Characters are inserted into the
  *    output sequence until one of the following occurs:
  *
  *        exactly count characters are inserted
  *        inserting into the output sequence fails (in which case
  *        setstate(badbit) is called).
  */
    }
}

void save_floats(const std::vector<float> &sig, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary | APPEND_FLAG);
    for (auto &elt: sig)
    {
        ofs.write(reinterpret_cast<const char*> (&elt), sizeof(float));;
    }
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

std::vector<float> phase_diff_wrapped(const std::vector<cf32> &iq)
{
    std::vector<float> angle_diff (iq.size(), 0);

    for (auto [e, elt]: std::views::enumerate(iq))
    {
        if (e == 0) {continue;}
        float diff = std::arg(iq[e]) - std::arg(iq[e-1]);
        if (diff < -PI) {diff += PI;}
        if (diff >  PI) {diff -= PI;}
        // printf("%+3.2f - %+3.2f = %+2.2f \n",  std::arg(iq[e]) * RAD2DEG, std::arg(iq[e-1]) * RAD2DEG, diff * RAD2DEG);
        angle_diff[e] = diff;
    }

    return angle_diff;
}
