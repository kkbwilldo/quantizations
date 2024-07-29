#ifndef ops_H
#define ops_H

#include <cuda_fp16.h>

typedef enum DataType_t
{
  General8bit = 0,
  FP4 = 1,
} DataType_t;


void quantize(float *code, float *A, unsigned char *out, int n);
void dequantize(float *code, unsigned char *A, float *out, int n);
template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int block_size, const int n);

template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits);
template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize);
template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize);

#endif
