CXXFLAGS=-Wall -Wextra
CPPSTD=c++17
LIBS=-lrtlsdr -lm

.PHONY: all clean

all: main

%: %.cpp
	clang++ $(CXXFLAGS) -std=$(CPPSTD) $< -o $@ $(LIBS)
	./$@ out.iq

inspect:
	inspectrum --format sc8 *iq

sharp:
	clang-format -i *.cpp
