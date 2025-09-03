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
#define DEFAULT_SR          2'400'000   // 2.4 MSPS
#define DEFAULT_GAIN        0           // auto-gain
#define READ_SIZE           0x01 << 18  // 262,144 samples

#define FULL_SCALE_AUDIO    32768.f
#define NUM_AUDIO_CHAN      1
#define AUDIO_SR            48000
#define AUDIO_VOLUME        80.f/100.f

/* Convenience macros */
#define RED                 "\033[31m"
#define GREEN               "\033[32m"
#define BLUE                "\033[34m"
#define RESET               "\033[0m"
#define PI                  3.1415926535898f
#define RAD2DEG             180 / PI
#define DEG2RAD             PI / 180
// clang-format on

#define APPEND_FLAG std::ios::app // For fstream file writing operations

// Macros for colored fprintf
#define ERR_PRINT(fmt, ...) \
    fprintf(stderr, RED "[FATAL]" fmt RESET "\n", ##__VA_ARGS__)

#define INFO_PRINT(fmt, ...) \
    fprintf(stdout, GREEN "[INFO]" fmt RESET "\n", ##__VA_ARGS__)

std::vector<std::string> LINE_BREAKS = {"\t", "\t", "\t", "\t", "\t", "\n"};
size_t NUM_ELTS_PER_LINE = LINE_BREAKS.size();

#define ARR_PRINT(arr) \
    for (auto [e, elt] : std::views::enumerate(arr)) \
    printf(BLUE "(%+2.4f, %+2.4f)" \
                "%s" RESET, \
           elt.real(), elt.imag(), \
           LINE_BREAKS[e % NUM_ELTS_PER_LINE].c_str())

typedef std::complex<float> cf32;
/* --- */

// Access 1D vec as if it's a 2D array helper function
inline const float& vec_at(const std::vector<float> &arr, int r, int c, int ncols)
{
    return arr[r * ncols + c];
}

#define arr_at(arr_ptr, r, c, ncols) arr_ptr[r * ncols + c]

static float fir_taps[150] = {
    -8.891121979104355e-05, -4.568894291878678e-05, 1.324024321037531e-19,
    4.8910344048636034e-05, 0.00010176578507525846, 0.00015918372082524002,
    0.0002215989661635831,  0.0002891898329835385,  0.0003618094197008759,
    0.0004389254027046263,  0.0005195712437853217,  0.0006023115711286664,
    0.000685224134940654,   0.000765899836551398,   0.0008414634503424168,
    0.0009086140780709684,  0.0009636876056902111,  0.0010027388343587518,
    0.001021643984131515,   0.0010162197286263108,  0.0009823590517044067,
    0.0009161773486994207,  0.000814169819932431,   0.0006733705522492528,
    0.000491514103487134,   0.0002671921392902732,  -5.397885477439954e-19,
    -0.0003093295672442764, -0.0006588120013475418, -0.001045119483023882,
    -0.0014635164989158511, -0.0019078265177085996, -0.002370431087911129,
    -0.002842305926606059,  -0.003313093911856413,  -0.0037712145131081343,
    -0.004204011056572199,  -0.004597933497279882,  -0.004938754718750715,
    -0.005211817566305399,  -0.005402303300797939,  -0.005495528690516949,
    -0.005477248691022396,  -0.005333978217095137,  -0.00505330553278327,
    -0.004624209366738796,  -0.004037360195070505,  -0.0032853989396244287,
    -0.0023631977383047342, -0.0012680785730481148, 1.2666409924871375e-18,
    0.0014383041998371482,  0.0030412343330681324,  0.004800286144018173,
    0.006704068277031183,   0.008738373406231403,   0.01088631246238947,
    0.0131284911185503,     0.015443259850144386,   0.017806991934776306,
    0.020194420590996742,   0.0225790124386549,     0.02493337355554104,
    0.02722967229783535,    0.029440095648169518,   0.03153730556368828,
    0.03349488228559494,    0.03528778627514839,    0.03689277544617653,
    0.038288816809654236,   0.03945744410157204,    0.04038307070732117,
    0.04105329513549805,    0.041459083557128906,   0.041594959795475006,
    0.041459083557128906,   0.04105329513549805,    0.04038307070732117,
    0.03945744410157204,    0.038288816809654236,   0.03689277544617653,
    0.03528778627514839,    0.03349488228559494,    0.03153730556368828,
    0.029440095648169518,   0.02722967229783535,    0.02493337355554104,
    0.0225790124386549,     0.020194420590996742,   0.017806991934776306,
    0.015443259850144386,   0.0131284911185503,     0.01088631246238947,
    0.008738373406231403,   0.006704068277031183,   0.004800286144018173,
    0.0030412343330681324,  0.0014383041998371482,  1.2666409924871375e-18,
    -0.0012680785730481148, -0.0023631977383047342, -0.0032853989396244287,
    -0.004037360195070505,  -0.004624209366738796,  -0.00505330553278327,
    -0.005333978217095137,  -0.005477248691022396,  -0.005495528690516949,
    -0.005402303300797939,  -0.005211817566305399,  -0.004938754718750715,
    -0.004597933497279882,  -0.004204011056572199,  -0.0037712145131081343,
    -0.003313093911856413,  -0.002842305926606059,  -0.002370431087911129,
    -0.0019078265177085996, -0.0014635164989158511, -0.001045119483023882,
    -0.0006588120013475418, -0.0003093295672442764, -5.397885477439954e-19,
    0.0002671921392902732,  0.000491514103487134,   0.0006733705522492528,
    0.000814169819932431,   0.0009161773486994207,  0.0009823590517044067,
    0.0010162197286263108,  0.001021643984131515,   0.0010027388343587518,
    0.0009636876056902111,  0.0009086140780709684,  0.0008414634503424168,
    0.000765899836551398,   0.000685224134940654,   0.0006023115711286664,
    0.0005195712437853217,  0.0004389254027046263,  0.0003618094197008759,
    0.0002891898329835385,  0.0002215989661635831,  0.00015918372082524002,
    0.00010176578507525846, 4.8910344048636034e-05, 1.324024321037531e-19,
    -4.568894291878678e-05, -8.891121979104355e-05, 0.000000};
// Generated a 149 tap Hamming window 40dB suppression filter with GNU
// Radio and appended one zero to make it 150 taps. Now ready to hopefully
// use with polyphase resampling but also possibly not linear anymore.

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
void save_interleaved_cf32(const std::vector<cf32> &iq,
                           const std::string &filename);
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
    static const pa_sample_spec ss = {.format = PA_SAMPLE_S16LE,
                                      .rate = AUDIO_SR,
                                      .channels = NUM_AUDIO_CHAN};
    pa_simple *s = pa_simple_new(NULL, // Use default server
                                 "WFM Demod App", // Application Name
                                 PA_STREAM_PLAYBACK,
                                 NULL, // Use default device
                                 "La Musica", // Description of stream
                                 &ss, // Sample format
                                 NULL, NULL,
                                 NULL // ignore error code
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

    /* Play out through pulseaudio
     * We have 2^18 samples that we'll chunk into 2^10 sized sections
     * (Thereby giving us READ_SIZE / len_audio_buffer = 2^8=64 sections)
     * For each section we'll index into angle_diff at a different start and
     * end point so we get 1024 fresh sections in the buffer.
     *
     * We're recording at 2.4MSps, not playing back at the same rate
     * For now we'll do a 'naive' decimation where I'll just take one every
     * 50 every samples. Later, we'll want to polyphase resample by the same
     * decimation value.
     */
    size_t decimation_value = DEFAULT_SR / AUDIO_SR;

    size_t num_samples = angle_diff.size();
    const size_t len_audio_buffer = 0x01 << 11;
    int16_t audio_buffer[len_audio_buffer];

    int error;
    // Just to test, let's also write this audio_buffer out to a file
    // and see if it sounds like anything...
    std::ofstream ofs("audio_buffer.pcm", std::ios::binary | APPEND_FLAG);
    for (size_t i = 0; i < num_samples / (decimation_value * len_audio_buffer);
         i++)
    {
        int idx_start = i * len_audio_buffer * decimation_value;
        for (size_t j = 0; j < len_audio_buffer; j++)
        {
            int16_t val = static_cast<int16_t>(
                angle_diff[j * decimation_value + idx_start]
                * (AUDIO_VOLUME * FULL_SCALE_AUDIO) / PI);
            audio_buffer[j] = val;
            ofs.write(reinterpret_cast<const char *>(&val), sizeof(int16_t));
        }
        if (pa_simple_write(
                sdr_ctx->s, audio_buffer,
                len_audio_buffer * NUM_AUDIO_CHAN * sizeof(int16_t), &error)
            < 0)
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

void save_interleaved_cf32(const std::vector<cf32> &iq,
                           const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary | APPEND_FLAG);

    for (auto &elt : iq)
    {
        float i = elt.real();
        float q = elt.imag();

        ofs.write(reinterpret_cast<const char *>(&i), sizeof(float));
        ofs.write(reinterpret_cast<const char *>(&q), sizeof(float));
        /*     basic_ostream& write( const char_type* s, std::streamsize count
         * ); s: char pointer to the first byte of the variable's address
         *     count: the number of bytes to copy over starting from the
         * address pointed by s
         *
         *    outputs the characters from successive locations in the character
         * array whose first element is pointed to by s. Characters are
         * inserted into the output sequence until one of the following occurs:
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
    for (auto &elt : sig)
    {
        ofs.write(reinterpret_cast<const char *>(&elt), sizeof(float));
        ;
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
    std::vector<float> angle_diff(iq.size(), 0);

    for (auto [e, elt] : std::views::enumerate(iq))
    {
        if (e == 0)
        {
            continue;
        }
        float diff = std::arg(iq[e]) - std::arg(iq[e - 1]);
        if (diff < -PI)
        {
            diff += PI;
        }
        if (diff > PI)
        {
            diff -= PI;
        }
        // printf("%+3.2f - %+3.2f = %+2.2f \n",  std::arg(iq[e]) * RAD2DEG,
        // std::arg(iq[e-1]) * RAD2DEG, diff * RAD2DEG);
        angle_diff[e] = diff;
    }

    return angle_diff;
}

void polyphase_resample(const float* fir, const std::vector<float> &x, float[2622] y)
{
    /*
     * Resamples before implementing an FIR filter which is better for
     * performance.
     * Following this guide:
     *                  https://www.dsprelated.com/showarticle/191.php
     *
     * To make indexing easier, After the convolution outputs I'll split
     * the first row and all the rest into two separate arrays.
     */

    // TODO: don't hardcode decimation value
    std::vector<std::vector<float>> sections(50);

    sections[0] = conv(x[0], h[0]);  // The first one is handled differently
    for (size_t conv_num = 1; conv_num < 50; conv_num++)
    {
        sections[conv_num] = conv(x[50-conv_num], fir[conv_num]);
    }

    y[0] = sections[0][0];
    for (size_t i = 1; i < 2622; i++)
    {
        sum = sections[0][i];  // accumulate this for y[i]
        for (size_t j = 0; j < 50-1; j++)
        {
            sum += sections[j][i-1];
        }
        y[i] = sum;
    }
}

std::vector<float> conv(const std::vector<float> &x, const std::vector<float> &h)
{
    /*
     * Hand-rolling a convolution function because why not add the worry of not
     * doing this section correctly too.
     *
     * Won't bother profiling but should probably consider doing the FFT
     * version of this in case it's faster. But in our use case, we're
     * convolving 1x3 elt arrays with 1x2622 elt arrays. Probably not worth it.
     *
     * According to GPT, I can probably use vectorization if my compiler
     * can understand what I'm trying to say...
     *
     * Rules:   Pass sizes explicitly
     *          Mark inputs const and outputs non-const; use restrict to
                declare no aliasing b/n them
     *          Loop has no branches and has no history of previous results
     *          No function calls inside hot loops unless inlinable
     *          Operate on contiguous data and avoid stride/weird increments
     *
     *          Ask the compiler to tell you:
                    GCC: -fopt-info-vec-optimized -fopt-info-vec-missed
                    Clang: -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
     */

    // Plan:
    // For N = x.size() and M = h.size()
    // -- make a (NxM-1)x(NxM-1) sized 2D float array of zeros
    // -- for the X vector (my input samples), go through the array
    //

    // Naive approach:
    size_t N = x.size();
    size_t M = h.size();
    size_t h_pad_size = M + 2 * N - 2;

    std::vector<float> h_pad(h_pad_size, 0);
    std::vector<float> y(N, 0);

    std::copy(h.begin(), h.begin() + M, // src start and end slices
              h_pad.begin() + N - 1); // zero pad b by N-1
                                      // zeros on either side

    std::reverse(h_pad.begin(), h_pad.end()); // reverse second arr before conv

    for (size_t i = 0; i < N; i++)
    {
        int start = h_pad_size - N - i;
        for (size_t j = 0; j < N; j++)
        {
            y[i] += x[j] * h_pad[start + j];
        }
    }

    return y;
}
