CXXFLAGS=-Wall -Wextra
CPPSTD=c++11
%: %.cpp
	clang++ $(CXXFLAGS) -std=$(CPPSTD) $< -o $@ -lm
	./$@
