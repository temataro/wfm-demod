#include <array>
#include <cstdio>
#include <string>
#include <complex>
#include <iostream>
// #include <vector>

typedef std::complex<float> cf32;
constexpr int N = 0x00010000;  // number of samples in buffer

// === prototype jail ===
template <std::size_t SIZE>
void view_array(const std::array<cf32, SIZE> iq);
std::array<cf32, N> read_iq(const std::string& filename, int *offset);
//    ==============

template <std::size_t SIZE>
void view_array(const std::array<cf32, SIZE>& iq)
{
    for (int i = 0; i < iq.size(); i++)
    {
        printf("(%.2f + j%.2f), ", iq[i].real(), iq[i].imag());
    }
}


std::array<cf32, N> read_iq(std::string& filename, int *offset)
{
    // Read IQ data from a file in 65K chunks.
    // Offset by N so it can
    *offset += N;
    printf("offset updated to %d.\n", *offset);
    printf("Filename: %s\n", filename.c_str());
    std::array<cf32, N> iq = {0};

    return iq;
}

int main()
{
    int offset = 0;
    std::string filename = "filename.iq";

    std::array<cf32, N> iq = read_iq(filename, &offset);
    printf("iq.size: %zuK elements. \n",(size_t) iq.size() / 1000);

    std::array<cf32, N> iq2 = read_iq(filename, &offset);
    printf("iq2.size: %zuK elements. \n",(size_t) iq2.size() / 1000);

    return 0;
}
