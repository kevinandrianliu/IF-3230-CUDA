all: ./src/main.cu
	nvcc ./src/main.cu -o radix_sort

clean:
	$(RM) radix_sort
