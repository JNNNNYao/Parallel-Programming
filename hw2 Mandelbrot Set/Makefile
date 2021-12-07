CC = clang
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O3 -march=native
hw2a: CFLAGS += -pthread -ffast-math
hw2a_time: CFLAGS += -pthread -ffast-math
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math -ffinite-math-only -fno-rounding-math -fno-signaling-nans -fcx-limited-range
hw2b_time: CC = mpicc
hw2b_time: CXX = mpicxx
hw2b_time: CFLAGS += -fopenmp -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math -ffinite-math-only -fno-rounding-math -fno-signaling-nans -fcx-limited-range
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b hw2a_time hw2b_time

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
