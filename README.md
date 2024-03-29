# CPU vs GPU Convolution and Normalization
## ECE 7720 HW 3

### Part 1
1. CPU convolution over input array size of 10,000,000 and impulse length of 2001 took on average 59177.4 milliseconds. CPU normalization over the same input took on average 53.56 milliseconds.
2. Each element of the output of convolution can be calculated in parallel because there is no data dependency between elements. Finding the maximum for normalization can be reduced using multiple threads but not fully parallelized. Division for normalization is an element-wise calculation and can be fully parallelized.

### Part 2A
1. The table of results can be found in results/results.csv (open it in excel or numpy.genfromtxt or just look at the figures).
2. Because each element can be calculated in parallel, each GPU thread is assigned a set of elements to compute. If the number of threads exceeds the number of elements then the algorithm is parallelized very well. If the number of threads is less than the number of elements, each thread operates on a set of elements until all have been calculated. Each thread, as it computes an element, loops for the length of the impulse and computes the sum of the products of successive elements in the input and impulse (convolution). If dynamic parallelism were allowed in this assignment, the sum portion of the convolution could be reduced as well. This would likely provide a large speedup for large impulse lengths.
3. The results show that there is less of a correlation between # threads per block and speedup and # blocks and speedup than just # total threads and speedup. My implementation does not take advantage of block shared memory and therefore is not impacted greatly by # of threads per block specifically. If dynamic parallelism were allowed, shared memory could be used to reduce the sum portion of convolution and this would likely result in an additional speedup. # total threads has a significant effect on speedup as shown in results/speedup.png and results/conv_time.png.
4. The 256x256 configuration results in a speedup factor of 84.14. When changed to 1024x1024 the speedup factor is 99.19. Though both speedup factors are great, the jump is also significant and expected due to the parallelized algorithm used and the increase in total threads.
5. Assigning fewer threads/blocks (total threads) is sub-optimal because it doesn't take advantage of the full potential of the parallelized algorithm. The algorithm performs optimally when the total number of threads is equal to the number of elements in the output such that each element is computed simultaneously.

### Part 2B
1. The table is found in results/results.csv (excel, numpy.genfromtxt, text editor).
2. Finding the maximum value in an array can be reduced using a GPU by finding the maximum value of chunks simultaneously then finding the maximum of each sub-problem recursively. Division is an element-wise operation and is therefore fully parallelizable. I chose to divide the input array into chunks such that each thread computes the local max in the chunk then uses the atomicMax function to serialize the operation of each thread into the global max. When the number of threads is equal to the input size, the algorithm used is completely serial. There is probably an optimal number of threads for reducing the max using the algorithm I chose. If dynamic parallelism were allowed in this assignment, the max could be found utilizing shared memory and consistently reducing the problem from O(N) to O(log(N)). As for the division, it performs optimally when the number of threads is equal to the input size.
3. As seen in results/speedup.png and results/norm_time.png the max speedup achieved is at 1024x1024. The large speedup increase is when the configuration changes from 8x8 to 256x256 but remains pretty constant after that.
4. Explained in the last question. There is very little increase from 256x256 to 1024x1024. I believe this is due to an increase in speedup during division but a slight decrease from the maximum kernel because 1024x1024 (1,048,576) threads must be serialized through the atomicMax used at the end.
5. I believe that max and divide are somewhat inversely proportional in speedup as thread count increases as explained in the last question. The chosen max algorithm has a sweet spot close to where the chunk size matches the number of threads so that serialization at the end doesn't take too much time. Divide on the other hand only performs better as thread count increases.

![Time analysis of convolution: CPU vs GPU](/results/conv_time.png?raw=true)
![Time analysis of normalization: CPU vs GPU](/results/norm_time.png?raw=true)
![Speedup analysis](/results/speedup.png?raw=true)
