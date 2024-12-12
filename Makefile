.PHONY: all clean

CC=clang
CXX=clang++
CXXFLAGS=-stdlib=libc++
LDFLAGS=-fuse-ld=lld
SUFFIX=

all: build/lbtbench$(SUFFIX)

clean:
	$(RM) -r build
	$(MAKE) -C libblastrampoline/src clean

build/lbtbench$(SUFFIX): lbtbench.cpp libblastrampoline/src/build/libblastrampoline.so.5
	mkdir -p build
	$(CXX) -o $@ -Ilibblastrampoline/include/ILP64/x86_64-linux-gnu -Ilibblastrampoline/src -O3 '-Wl,-rpath=$$ORIGIN/../libblastrampoline/src/build' -ffast-math -flto -fopenmp -g -std=gnu++20 $(CXXFLAGS) $< libblastrampoline/src/build/libblastrampoline.so.5 $(LDFLAGS)

libblastrampoline/src/build/libblastrampoline.so.5:
	$(MAKE) -C libblastrampoline/src 'CC=$(CC)' 'LDFLAGS=$(LDFLAGS)'
