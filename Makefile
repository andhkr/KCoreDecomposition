NVCC = nvcc

CU_SRC =  CPUfixedKCore.cu GPUcudaFixedKCore.cu Main.cu
HDR    = FixedKCore.cuh
TARGET = fixedKCoreComputation

ARGS ?= 100000 499985 testGraph_100000_499985.txt 6

all: $(TARGET)

$(TARGET): $(CU_SRC)  $(HDR)
	$(NVCC) -rdc=true $(CU_SRC)  -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET) *.o
