CXXFLAGS=-Wall -Wextra
CPPSTD=c++23
LIBS=-lrtlsdr -lm -lpulse-simple -lpulse
OPTS=-march=native -ffast-math -O2

.PHONY: all clean

all: radio

%: %.cpp
	clang++ $(CXXFLAGS) -std=$(CPPSTD) $< -o $@ $(LIBS) $(OPTS)
	./$@ out.iq

inspect:
	inspectrum *cf32
	inspectrum --format sc8 *iq

graph:
	# make the interactive plot with average processing time
	gnuplot -e "set term qt font 'DejaVu Sans,12'; \
		set title 'Processing delay in cb (in ns)' font ',14'; \
		set xlabel 'call #' font ',12'; \
		set ylabel 'proc_time (ns)' font ',12'; \
		set grid; \
		set key left top; \
		stats 'cb_bench.txt' using (real(strcol(1))) name 'S' nooutput; \
		mean=S_mean; \
		mn=S_min; \
		mx=S_max; \
		set style line 1 lt 1 lw 2 pt 7 ps 1.2 linecolor rgb '#008080'; \
		set style line 2 lt 0 lw 3 linecolor rgb '#ff7f0e'; \
		set style line 3 lt 0 lw 4 linecolor rgb '#d62728'; \
		plot 'cb_bench.txt' using (column(0)+1):(real(strcol(1))) with linespoints ls 1 title 'data', mean with lines ls 2 title sprintf('mean = %.6g ns', mean), mn with lines ls 3 title sprintf('min = %.6g ns', mn), mx with lines ls 3 title sprintf('max = %.6g ns', mx); \
		pause -1 'Press Enter to close'"

	# save to png
	gnuplot -e "set terminal pngcairo size 1200,800 enhanced font 'DejaVu Sans,14'; \
		set output 'plot.png'; \
		set title 'Processing delay in cb (in ns)' font ',14'; \
		set xlabel 'call #' font ',12'; \
		set ylabel 'proc_time (ns)' font ',12'; \
		set grid; \
		set key left top; \
		stats 'cb_bench.txt' using (real(strcol(1))) name 'S' nooutput; \
		mean=S_mean; \
		mn=S_min; \
		mx=S_max; \
		set style line 1 lt 1 lw 2 pt 7 ps 1.2 linecolor rgb '#008080'; \
		set style line 2 lt 0 lw 3 linecolor rgb '#ff7f0e'; \
		set style line 3 lt 0 lw 4 linecolor rgb '#d62728'; \
		plot 'cb_bench.txt' using (column(0)+1):(real(strcol(1))) with linespoints ls 1 title 'data', mean with lines ls 2 title sprintf('mean = %.6g ns', mean), mn with lines ls 3 title sprintf('min = %.6g ns', mn), mx with lines ls 3 title sprintf('max = %.6g ns', mx); \
		set output"


sharp:
	clang-format -i *.cpp

clean:
	rm -f *cf32 *wav *sc8 *cs8 *f32 *pcm
