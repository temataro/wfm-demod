CXXFLAGS=-Wall -Wextra
CPPSTD=c++14
LIBS=-lrtlsdr -lm

.PHONY: all clean

all: main

%: %.cpp
	clang++ $(CXXFLAGS) -std=$(CPPSTD) $< -o $@ $(LIBS)
	./$@ out.iq

sharp:
	clang-format -i *.cpp
