## Using `gnuplotpp.hpp`

When doing algorithm prototyping in Python it's easy to quickly plot
out Numpy arrays and get a feel for where your data is going wrong.
No external C++ libraries.

* Talks to gnuplot via popen().

* Supports interactive (qt/wxt) and file output (PNG/SVG).

API:
---

    * Gnuplot,

    * set_title/xlabel/ylabel/grid/limits,

    * plot_line,

    * plot_line_add,

    * scatter,

    * imshow.

### Usage

```cpp
// main.cpp
#include "gnuplotpp.hpp"
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    // Choose interactive or file mode based on CLI
    //   ./demo          -> interactive (qt)
    //   ./demo png out.png  -> PNG file
    //   ./demo svg out.svg  -> SVG file
    std::string term = "qt", out;
    if (argc == 3 && std::string(argv[1]) == "png") { term="png"; out=argv[2]; }
    if (argc == 3 && std::string(argv[1]) == "svg") { term="svg"; out=argv[2]; }

    gpp::Gnuplot gp(term, out);

    // Example: lines
    const int N = 800;
    std::vector<double> x(N), s(N), c(N);
    for (int i = 0; i < N; ++i) {
        double t = i * 0.01;
        x[i] = t; s[i] = std::sin(t); c[i] = std::cos(t);
    }
    gp.set_title("sin vs cos");
    gp.set_xlabel("x"); gp.set_ylabel("y"); gp.set_grid(true);
    gp.plot_line(x, s, "sin(x)", "with lines lw 2 lc rgb 'blue'");
    gp.plot_line_add(x, c, "cos(x)", "with lines lw 2 lc rgb 'red'");

    // Example: scatter
    std::vector<double> xs, ys;
    for (int i = 0; i < 60; ++i) { double t = i * 0.2; xs.push_back(t); ys.push_back(std::sin(t)+0.1*std::cos(8*t)); }
    gp.scatter_add(xs, ys, "samples", "with points pt 7 ps 1.5 lc rgb 'black'");

    // Example: heatmap
    const int H = 50, W = 70;
    std::vector<std::vector<double>> Z(H, std::vector<double>(W));
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            Z[i][j] = std::sin(i*0.2) * std::cos(j*0.2);
    gp.imshow(Z, "Heatmap");

    return 0;
}
```

### Building
```bash
# deps
sudo apt update
sudo apt install -y clang gnuplot gnuplot-qt

# build
clang++ -std=gnu++17 main.cpp -o demo

# interactive window
./demo

# save to PNG (no window)
./demo png out.png

# save to SVG (no window)
./demo svg out.svg
```
