/*
 * This header file was written with GPT5.
 * Learn how to use it in the README.md file.
 */

#ifndef GNUPLOTPP_HPP
#define GNUPLOTPP_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cstdarg>

namespace gpp {

class Gnuplot {
public:
    explicit Gnuplot(const std::string& term = "qt", const std::string& outfile = "")
    : gp_(nullptr), term_(term), outfile_(outfile) {
        gp_ = popen("gnuplot -persist", "w");
        if (!gp_) throw std::runtime_error("Failed to start gnuplot (is it installed?)");

        if (!outfile_.empty()) {
            if (term_ == "png") {
                cmd("set term pngcairo size 1000,700 enhanced font ',10'");
                cmd("set output '%s'", outfile_.c_str());
            } else if (term_ == "svg") {
                cmd("set term svg size 1000,700 dynamic");
                cmd("set output '%s'", outfile_.c_str());
            } else {
                cmd("set term pngcairo size 1000,700 enhanced font ',10'");
                cmd("set output '%s'", outfile_.c_str());
            }
        } else {
            if (term_ == "qt") cmd("set term qt size 1000,700");
            else if (term_ == "wxt") cmd("set term wxt size 1000,700");
            else cmd("set term qt size 1000,700");
        }

        cmd("set key opaque");
        cmd("set grid");
        cmd("set palette rgb 7,5,15"); // pleasant default
    }

    // non-copyable (owns a pipe), but movable if you want—keep simple for now
    Gnuplot(const Gnuplot&) = delete;
    Gnuplot& operator=(const Gnuplot&) = delete;

    ~Gnuplot() {
        if (gp_) {
            if (!outfile_.empty()) cmd("unset output");
            pclose(gp_);
        }
    }

    void set_title(const std::string& t)  { cmd("set title %s", quote(t).c_str()); }
    void set_xlabel(const std::string& l) { cmd("set xlabel %s", quote(l).c_str()); }
    void set_ylabel(const std::string& l) { cmd("set ylabel %s", quote(l).c_str()); }
    void set_grid(bool on = true)         { cmd(on ? "set grid" : "unset grid"); }
    void set_xlim(double a, double b)     { cmd("set xrange [%g:%g]", a, b); }
    void set_ylim(double a, double b)     { cmd("set yrange [%g:%g]", a, b); }

    // First series uses 'plot'. Add more with plot_line_add()/scatter_add() if you want.
    void plot_line(const std::vector<double>& x,
                   const std::vector<double>& y,
                   const std::string& title = "",
                   const std::string& style = "with lines lw 2") {
        ensure_same_size(x, y);
        const std::string block = new_block();
        begin_block(block);
        for (size_t i = 0; i < x.size(); ++i) std::fprintf(gp_, "%.17g %.17g\n", x[i], y[i]);
        end_block();
        emit_plot(block, title, style, /*replot=*/false);
    }

    void plot_line_add(const std::vector<double>& x,
                       const std::vector<double>& y,
                       const std::string& title = "",
                       const std::string& style = "with lines lw 2") {
        ensure_same_size(x, y);
        const std::string block = new_block();
        begin_block(block);
        for (size_t i = 0; i < x.size(); ++i) std::fprintf(gp_, "%.17g %.17g\n", x[i], y[i]);
        end_block();
        emit_plot(block, title, style, /*replot=*/true);
    }

    void scatter(const std::vector<double>& x,
                 const std::vector<double>& y,
                 const std::string& title = "",
                 const std::string& style = "with points pt 7 ps 1.5") {
        ensure_same_size(x, y);
        const std::string block = new_block();
        begin_block(block);
        for (size_t i = 0; i < x.size(); ++i) std::fprintf(gp_, "%.17g %.17g\n", x[i], y[i]);
        end_block();
        emit_plot(block, title, style, /*replot=*/false);
    }

    void scatter_add(const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::string& title = "",
                     const std::string& style = "with points pt 7 ps 1.5") {
        ensure_same_size(x, y);
        const std::string block = new_block();
        begin_block(block);
        for (size_t i = 0; i < x.size(); ++i) std::fprintf(gp_, "%.17g %.17g\n", x[i], y[i]);
        end_block();
        emit_plot(block, title, style, /*replot=*/true);
    }

    // Matrix -> image (heatmap). Z[row][col]
    void imshow(const std::vector<std::vector<double>>& Z,
                const std::string& title = "Heatmap",
                bool keep_aspect = true) {
        if (Z.empty() || Z[0].empty()) throw std::runtime_error("imshow: empty matrix");
        size_t rows = Z.size(), cols = Z[0].size();
        for (const auto& r : Z) if (r.size() != cols) throw std::runtime_error("imshow: ragged matrix");

        cmd("set view map");
        cmd(keep_aspect ? "set size ratio -1" : "set size noratio");
        cmd("unset key");

        const std::string block = new_block();
        cmd("%s matrix << EOD", block.c_str());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) std::fprintf(gp_, "%.17g ", Z[i][j]);
            std::fprintf(gp_, "\n");
        }
        std::fprintf(gp_, "EOD\n");

        cmd("set title %s", quote(title).c_str());
        cmd("plot %s matrix with image", block.c_str());
    }

private:
    FILE* gp_;
    std::string term_, outfile_;
    int block_counter_ = 0;

    void cmd(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        std::vfprintf(gp_, fmt, args);
        std::fprintf(gp_, "\n");
        std::fflush(gp_);
        va_end(args);
    }

    static void ensure_same_size(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) throw std::runtime_error("x and y must have same length");
    }

    std::string new_block() {
        std::ostringstream ss; ss << "$B" << (++block_counter_); return ss.str();
    }

    void begin_block(const std::string& name) { cmd("%s << EOD", name.c_str()); }
    void end_block() { std::fprintf(gp_, "EOD\n"); std::fflush(gp_); }

    void emit_plot(const std::string& block, const std::string& title,
                   const std::string& style, bool replot) {
        std::ostringstream pl;
        pl << (replot ? "replot " : "plot ") << block << " using 1:2 " << style;
        if (!title.empty()) pl << " title " << quote(title); else pl << " notitle";
        cmd("%s", pl.str().c_str());
    }

    static std::string quote(const std::string& s) {
        std::ostringstream q; q << "'";
        for (char c : s) q << (c=='\'' ? "\\'" : std::string(1,c));
        q << "'"; return q.str();
    }
};

} // namespace gpp

#endif // GNUPLOTPP_HPP

