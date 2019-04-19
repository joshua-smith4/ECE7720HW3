/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    const unsigned int padded_length,
    const unsigned int impulse_len) 
{
    
    int numThreads = gridDim.x*blockDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    while(idx < padded_length)
    {
	    float sumReal = 0.0;
	    float sumImg = 0.0;
	    int impulseIdx = 0;
    	for(int j = idx; j >= 0 && impulseIdx < impulse_len; --j)
	    {
            sumReal += (raw_data[j].x*impulse_v[impulseIdx].x-raw_data[j].y*impulse_v[impulseIdx].y);
            sumImg += (raw_data[j].x*impulse_v[impulseIdx].y+raw_data[j].y*impulse_v[impulseIdx].x);
	        ++impulseIdx; 
        }
        out_data[idx].x = sumReal; 
        out_data[idx].y = sumImg;
	    idx += numThreads;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    int numThreads = gridDim.x*blockDim.x;
    int sizeOfChunk = (padded_length+numThreads-1)/numThreads;
    int idx = (blockDim.x*blockIdx.x + threadIdx.x)*sizeOfChunk;
    int iter = 0;
    float local_max = out_data[0].x;
    while(idx+iter < padded_length && iter < sizeOfChunk)
    {
        if(out_data[idx+iter].x > local_max) local_max = out_data[idx+iter].x;
        ++iter;
    }  
    atomicMax(max_abs_val,local_max);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    int numThreads = gridDim.x*blockDim.x;
    int sizeOfChunk = (padded_length+numThreads-1)/numThreads;
    int idx = (blockDim.x*blockIdx.x + threadIdx.x)*sizeOfChunk;
    int iter = 0;
    float max_val_modified = 0.99999/(*max_abs_val);
    while(idx+iter < padded_length && iter < sizeOfChunk)
    {
        out_data[idx+iter].x *= max_val_modified;
        ++iter;
    }  
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length,
        const unsigned int impulse_len) {
        

    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks,threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length, impulse_len);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks,threadsPerBlock>>>(out_data,max_abs_val,padded_length);    
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks,threadsPerBlock>>>(out_data,max_abs_val,padded_length);
}
