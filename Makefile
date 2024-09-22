# Define compilers
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS = -std=c++11 -O3

# CUDA paths
CUDA_PATH = /usr/local/cuda
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

# Source files
CPP_SOURCES = $(wildcard src/*.cpp)
CUDA_SOURCES = src/kmeans_kernel.cu

# Object files
OBJECTS = $(CPP_SOURCES:.cpp=.o) src/kmeans_kernel.o

# Target executable
TARGET = bin/my_program

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Rule to compile C++ files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile CUDA files into .o files
src/kmeans_kernel.o: $(CUDA_SOURCES)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)
