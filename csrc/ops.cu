#include "ops.cuh"
#include "kernels.cuh"

/**
 * ops.cu는 커널을 호출하는 함수들을 정의 
 */


void quantize(float *code, float *A, unsigned char *out, int n)
{
    /**
     * @brief 배열을 양자화하는 함수.
     *
     * quantize는 주어진 배열 A를 양자화하여 결과를 out 배열에 저장합니다.
     * 양자화를 수행하기 위해 kQuantize 커널을 호출합니다.
     *
     * @param code 양자화 코드북에 대한 포인터.
     * @param A 양자화할 입력 배열에 대한 포인터.
     * @param out 양자화된 결과를 저장할 출력 배열에 대한 포인터.
     * @param n 입력 배열 A의 요소 수.
     */

    // 1024개의 스레드로 구성된 블록 수를 계산
    int num_blocks = n/1024;
    // n이 1024의 배수가 아닌 경우, 블록 수를 하나 더 추가
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    // kernels.cu의 kQuantize 커널 호출
    kQuantize<<<num_blocks, 1024>>>(code, A, out, n);
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
    /**
     * @brief 양자화된 배열을 복원하는 함수.
     *
     * dequantize는 주어진 양자화된 배열 A를 복원하여 결과를 out 배열에 저장합니다.
     * 복원 작업을 수행하기 위해 kDequantize 블록을 호출합니다.
     *
     * @param code 복원에 사용할 코드북에 대한 포인터.
     * @param A 양자화된 입력 배열에 대한 포인터.
     * @param out 복원된 결과를 저장할 출력 배열에 대한 포인터.
     * @param n 입력 배열 A의 요소 수.
     */

    // 1024개의 스레드로 구성된 블록 수를 계산
    int num_blocks = n/1024;
    // n이 1024의 배수가 아닌 경우, 블록 수를 하나 더 추가
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    // kernels.cu의 kDequantize 커널 호출
    kDequantize<<<num_blocks, 1024>>>(code, A, out, n);
}

template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
    /**
     * @brief 블록 단위로 배열을 양자화하는 함수.
     *
     * quantizeBlockwise는 주어진 배열 A를 블록 단위로 양자화하여 결과를 out 배열에 저장합니다.
     * 양자화를 수행하기 위해 kQuantizeBlockwise 커널을 호출합니다.
     *
     * @tparam T 데이터 타입 (예: float, half).
     * @tparam STOCHASTIC 양자화 방법 (확률적 양자화 여부).
     * @tparam DATA_TYPE 데이터 타입의 식별자.
     *
     * @param code 양자화 코드북에 대한 포인터.
     * @param A 양자화할 입력 배열에 대한 포인터.
     * @param absmax 각 블록의 절대 최대값에 대한 포인터.
     * @param out 양자화된 결과를 저장할 출력 배열에 대한 포인터.
     * @param rand 확률적 양자화를 위한 난수 배열에 대한 포인터.
     * @param rand_offset 난수 배열의 오프셋.
     * @param blocksize 블록 크기.
     * @param n 입력 배열 A의 요소 수.
     */

    // 주어진 배열의 원소 수에 따라 필요한 블록 수를 계산
    int num_blocks = n/blocksize;
    // n이 blocksize의 배수가 아닌 경우, 블록 수를 하나 더 추가
    num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;

    // blocksize에 따라 다른 kQuantizeBlockwise 커널을 호출
    if(blocksize == 4096)
        kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, DATA_TYPE><<<num_blocks, 1024>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 2048)
        kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE><<<num_blocks, 512>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 1024)
        kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 512)
        kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 256)
        kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE><<<num_blocks, 128>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 128)
        kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE><<<num_blocks, 64>>>(code, A, absmax, out, rand, rand_offset, n);
    else if(blocksize == 64)
        kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, rand, rand_offset, n);
}

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
    /**
     * @brief 블록 단위로 양자화된 배열을 복원하는 함수.
     *
     * dequantizeBlockwise는 주어진 양자화된 배열 A를 블록 단위로 복원하여 결과를 out 배열에 저장합니다.
     * 복원 작업을 수행하기 위해 kDequantizeBlockwise 커널을 호출합니다.
     *
     * @tparam T 데이터 타입 (예: float, half).
     * @tparam DATA_TYPE 데이터 타입의 식별자.
     *
     * @param code 복원에 사용할 코드북에 대한 포인터.
     * @param A 양자화된 입력 배열에 대한 포인터.
     * @param absmax 각 블록의 절대 최대값에 대한 포인터.
     * @param out 복원된 결과를 저장할 출력 배열에 대한 포인터.
     * @param blocksize 블록 크기.
     * @param n 입력 배열 A의 요소 수.
     */

    // 주어진 배열의 원소 수에 따라 필요한 블록 수 계산
    int num_blocks = n/blocksize;
    // n이 blocksize의 배수가 아닌 경우, 블록 수를 하나 더 추가
    num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
    
    int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

    // kDequantizeBlockwise 커널 호출
    if(DATA_TYPE > 0)
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize/2, n);
    else
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize, n);
}


template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits)
{
    /**
     * @brief 행렬 곱셈을 수행하는 함수.
     *
     * gemm_host는 주어진 행렬 A와 B의 곱셈을 수행하여 결과를 out 행렬에 저장합니다.
     * 곱셈 작업을 수행하기 위해 gemm_device 커널을 호출합니다.
     *
     * @tparam T 데이터 타입 (예: float, half).
     *
     * @param m 행렬 A의 행 수 및 행렬 C의 행 수.
     * @param n 행렬 B의 열 수 및 행렬 C의 열 수.
     * @param k 행렬 A의 열 수 및 행렬 B의 행 수.
     * @param A 행렬 A에 대한 포인터.
     * @param B 행렬 B에 대한 포인터.
     * @param out 결과 행렬 C에 대한 포인터.
     * @param lda 행렬 A의 leading dimension.
     * @param ldb 행렬 B의 leading dimension.
     * @param ldc 행렬 C의 leading dimension.
     * @param bits 연산에 사용할 비트 수.
     */

    // 필요한 블록 수 계산
    int num_blocks = (m+31)/32;

    // 16 비트인 경우 gemm_device 커널을 호출
    if(bits == 16)
        gemm_device<T, 16, 160><<< num_blocks, 160, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
}

template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize)
{
    int num_blocks = (m+31)/32;
    kgemm_4bit_inference<T, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
}

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{
    int num_blocks = (m+3)/4;
    kgemm_4bit_inference_naive<T, 128, BITS><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}


// template 인스턴스
template void gemm_4bit_inference<half>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<half, 16>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<__nv_bfloat16, 16>(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<float, 32>(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

template void gemm_host<half>(int m, int n, int k, half * A,  half* B,  half * out,  int lda, int ldb, int ldc, int bits);

template void quantizeBlockwise<half, 1, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, FP4>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, FP4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 1, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, FP4>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);

template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<half, General8bit>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<half, FP4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, FP4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);


