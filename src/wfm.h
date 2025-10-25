/*
wfm.h
*/
#include <vector>
#include <stdio.h>

// clang-format off
namespace defaults
{
    constexpr uint16_t DEFAULT_FC       = 106'000'000; // 106 MHz (Radio Two)
    constexpr uint16_t DEFAULT_SR       = 2'400'000; // 2.4 MSPS
    constexpr uint16_t DEFAULT_GAIN     = 0; // auto-gain
    constexpr uint16_t READ_SIZE        = 0x01 << 18; // 262;144 samples

    constexpr float FULL_SCALE_AUDIO    = 32768.f;
    constexpr float AUDIO_VOLUME        = 70.0f / 100.f; // 70%
    constexpr uint16_t NUM_AUDIO_CHAN   = 1;
    constexpr uint16_t AUDIO_SR         = 48'000;

    constexpr float PI                  = 3.1415926535898f;
} // namespace defaults
using namespace defaults;

/* Convenience macros */
#define RED                                "\033[31m"
#define GREEN                              "\033[32m"
#define BLUE                               "\033[34m"
#define RESET                              "\033[0m"

using cf32 = std::complex<float>;

#define APPEND_FLAG std::ios::app // For fstream file writing operations

// Macros for colored fprintf
#define ERR_PRINT(fmt, ...) \
    fprintf(stderr, RED "[FATAL]" fmt RESET "\n", ##__VA_ARGS__)

#define INFO_PRINT(fmt, ...) \
    fprintf(stdout, GREEN "[INFO]" fmt RESET "\n", ##__VA_ARGS__)

std::vector<std::string> LINE_BREAKS = {"\t", "\t", "\t", "\t", "\t", "\n"};
const size_t NUM_ELTS_PER_LINE = LINE_BREAKS.size();
const size_t decimation_value = DEFAULT_SR / AUDIO_SR;

#define ARR_PRINT(arr) \
    for (auto [e, elt] : std::views::enumerate(arr)) \
    printf(BLUE "(%+2.4f, %+2.4f)" \
                "%s" RESET, \
           elt.real(), elt.imag(), \
           LINE_BREAKS[e % NUM_ELTS_PER_LINE].c_str())

/* --- */

typedef struct
{
    uint64_t     sample_counter;
    clock_t      init_time;
    FILE         *fp;
    FILE         *benchmark_fp;
    pa_simple    *s;
    rtlsdr_dev_t *device;
} sdr_ctx_t;
// clang-format on
/* --- */

/* Prototype Jail */
void read_to_vec(unsigned char *buf, uint32_t len, std::vector<cf32> &iq);
std::vector<float> phase_diff_wrapped(const std::vector<cf32> &iq);
std::vector<float> conv(const std::vector<float> &x, const std::vector<float> &h);
std::vector<float> gptconvolve(const std::vector<float> &a, const std::vector<float> &b);
void save_floats(const std::vector<float> &sig, const std::string &filename);
void save_interleaved_cf32(const std::vector<cf32> &iq, const std::string &filename);
void test_convs();
/* --- */

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

std::vector<float> conv(const std::vector<float> &x,
                        const std::vector<float> &h)
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

    // zero pad h into h_pad
    for (size_t i = N; i < N + M; i++)
    {
        h_pad[i] = h[i - N];
    }

    std::reverse(h_pad.begin(), h_pad.end()); // reverse second arr before conv

    for (size_t j = 0; j < N; j++)
    {
        int start = h_pad_size - N - j;
        for (size_t k = 0; k < N; k++)
        {
            y[j] += x[k] * h_pad[start + k];
        }
    }

    return y;
}

std::vector<float> gptconvolve(const std::vector<float> &a,
                               const std::vector<float> &b)
{
    size_t n = a.size();
    size_t m = b.size();
    std::vector<float> result(n + m - 1, 0.0f);

    for (size_t u = 0; u < n; ++u)
        for (size_t v = 0; v < m; ++v)
            result[u + v] += a[u] * b[v];

    return result;
}

void test_convs()
{
    std::vector<float> x(12);
    std::vector<float> h(3);

    for (int i = 0; i < 12; i++)
    {
        x[i] = i;
        h[i % 3] = i;
    }

    std::vector<float> myconv = conv(x, h);
    std::vector<float> gptconv = gptconvolve(x, h);

    for (size_t i = 0; i < myconv.size(); i++)
    {
        printf("%.2f ", myconv[i]);
    }
    printf("\n\nGPT_CONV: \n\n");
    for (size_t i = 0; i < gptconv.size(); i++)
    {
        printf("%.2f ", gptconv[i]);
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

