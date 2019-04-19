# CPU vs GPU Convolution and Normalization
## ECE 7720 HW 3

### Part 1
1. CPU convolution over input array size of 10,000,000 and impulse length of 2001 took on average 59177.4 milliseconds. CPU normalization over the same input took on average 53.56 milliseconds.
2. Each element of the output of convolution can be calculated in parallel because there is no data dependency between elements. Finding the maximum for normalization can be reduced using multiple threads but not fully parallelized. Division for normalization is an element-wise calculation and can be fully parallelized.

### Part 2A
1. The table of results can be found in results/results.csv (open it in excel or numpy.genfromtxt or just look at the figures).
2. Because each element can be calculated in parallel, each GPU thread is assigned a set of elements to compute. If the number of threads exceeds the number of elements then the algorithm is parallelized very well. If the number of threads is less than the number of elements, each thread operates on a set of elements until all have been calculated. Each thread, as it computes an element, loops for the length of the impulse and computes the sum of the products of successive elements in the input and impulse (convolution). If dynamic parallelism were allowed in this assignment, the sum portion of the convolution could be reduced as well. This would likely provide a large speedup for large impulse lengths.
3. The results show that there is less of a correlation between # threads per block and speedup and # blocks and speedup than just # total threads and speedup. My implementation does not take advantage of block shared memory and therefore is not impacted greatly by # of threads per block specifically. If dynamic parallelism were allowed, shared memory could be used to reduce the sum portion of convolution and this would likely result in an additional speedup. # total threads has a significant effect on speedup as shown in results/speedup.png and results/conv_time.png.
4. 
