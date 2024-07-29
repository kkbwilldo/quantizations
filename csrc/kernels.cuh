#ifndef kernels
#define kernels

#include <float.h>
#include "ops.cuh"

__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n);
__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n);

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE> __global__ void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE> __global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n);

template <typename T, int BITS, int THREADS> __global__ void gemm_device(int M, int N, int K, T * __restrict__ const A,  T* B,  T * out,  int lda, int ldb, int ldc);
template <typename T, int THREADS> __global__ void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize);
template <typename T, int THREADS, int BITS> __global__ void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize);

#endif
