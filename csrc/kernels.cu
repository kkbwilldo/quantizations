#include "kernels.cuh"
#include <cub/cub.cuh>
#include <mma.h>

/**
 * T: 로드할 데이터 타입 - float, int, etc
 * BLOCK_SIZE : CUDA 블록 크기 
 * ITEMS_PER_THREAD : 각 스레드가 처리할 항목 수
 * cub::BLOCK_LOAD_WARP_TRANSPOSE : 데이터 로드 방식으로, warp 단위로 데이터를 로드하고 transpose하여 스레드에 할당
 * cub::BLOCK_STORE_WARP_TRANSPOSE : 데이터 저장 방식으로, warp 단위로 데이터를 저장하고 transpose하여 메모리에 배치
 * storage : CUB 라이브러리에서 임시 저장 공간으로 사용되는 메모리 <-- 이걸 shared mem에 선언하여 임시 메모리를 shared mem에서 관리
 * input_ptr : 입력 데이터 포인터
 * output_ptr : 출력 데이터 포인터
 * input_vals : 각 스레드가 저장할 값을 담은 배열
 * output_vals : 각 스레드가 로드한 값을 저장할 배열
 * valid_items : 유효한 원소 수로, 전체 데이터 크기를 초과하지 않도록 로드할 원소 수를 제한
 * 
 * 
 * cub::BlockLoad<T, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>(load_storage).Load(input_ptr, output_vals, valid_items);
 * cub::BlockStore<T, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>(sotre_storage).Store(output_ptr, input_vals, valid_items);
 * 로드와 저장 작업 후, 데이터를 일관성있게 처리하기 위해 '__syncthreads()'를 사용하여 스레드를 동기화
 */

#define TH 1024 // 최대 스레드 수
#define NUM 4
#define NUM_BLOCK 4096


/**
 * FP4 형식 설명.
 * 
 * FP4 형식은 다음과 같이 구성됩니다:
 * 부호 비트(sign)       (S): 1비트
 * 지수 비트(exponent)   (E): 2비트
 * 정밀도 비트(mantissa) (M): 1비트
 *
 * 부동 소수점 수는 다음과 같은 형태로 표현됩니다.
 * (-1)^(S) X 1.M X 2^{E-bias}  // 여기서 FP4는 바이어스로 3을 사용
 *
 * - 노멀 값(Normal values)
 *
 * 지수 비트 : 01, 10, 11 (지수 값이 최소가 아님)
 * 정밀도 비트 : 1이면 1.5, 0이면 1.0
 *
 * 지수 비트가 01인 경우, 실제 지수는 2^1 = 2
 * 지수 비트가 10인 경우, 실제 지수는 2^2 = 4
 * 지수 비트가 11인 경우, 실제 지수는 2^3 = 8
 *
 * M = 0인 경우 : 값 = 1.0 * 2^지수
 * M = 1인 경우 : 값 = 1.5 * 2^지수
 *
 * - 서브노멀 값(Subnormal values). 지수 비트가 0일 때
 *
 * 지수 비트 : 00
 * 정밀도 비트 : 1이면 최소 정밀도값, 0이면 0
 *
 * 지수 비트가 00인 경우, 실제 지수는 2^0 = 1
 * 
 * M = 0인 경우 : 값 = 0
 * M = 1인 경우 : 값 = 1 * 2^(-4) = 1/16 = 0.0625
 *
 *
 * bitsandbytes의 표현 값들은 제 기준으로는 이해가 되진 않습니다.(틀린 값을 쓰는 것 같은데,,,왜 잘 동작하는지,,,),
 * FP16까지는 IEEE 754 표준을 따라 정의하지만, FP4는 비표준 부동 소수점 형식이므로 bitsandbytes의 해당 값들을 차용해서 사용합니다.
 * issue reference : https://github.com/bitsandbytes-foundation/bitsandbytes/issues/988
 * issue reference : https://github.com/bitsandbytes-foundation/bitsandbytes/issues/851
 */


__device__ float dDequantizeFP4Tree(unsigned char val, float absmax)
{
    /**
     * @brief FP4 양자화된 값을 부동 소수점 값으로 디양자화합니다.
     *
     * dDequantizeFP4Tree는 FP4 형식의 양자화된 값과 최대 절대값을 사용하여 부동 소수점 값으로 변환합니다.
     * 비트 패턴을 사용하여 적절한 디양자화 값을 계산합니다.
     *
     * @param val 양자화된 FP4 값입니다.
     * @param absmax 최대 절대값입니다.
     * @return float 디양자화된 부동 소수점 값입니다.
     */

    float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f; // 부호 비트 추출

    // 지수 비트와 정밀도 비트를 사용해 값을 계산
    // 트리 구조의 첫 번째 분기 : 지수 비트의 첫 번째 비트를 확인
    if((val & 0b0100) == 4) // 0
        // 두 번째 분기 : 두 번째 지수 비트를 확인
        if((val & 0b0010) == 2) //01
            // 세번 째 분기 : 정밀 도 비트를 확인
            if((val & 0b0001) == 1) // 111
                return 0.25000000f*absmax*sign; // 1111
            else
                return 0.16666667f*absmax*sign; // 1110
        else
            if((val & 0b0001) == 1) // 110
                return 0.50000000f*absmax*sign; // 1101
            else
                return 0.33333333f*absmax*sign; // 1100
    else
        if((val & 0b0010) == 2) //10
            if((val & 0b0001) == 1) // 101
                return 1.00000000f*absmax*sign; // 1011
            else
                return 0.66666667f*absmax*sign; // 1010
        else
            if((val & 0b0001) == 1) // 100
                return 5.208333333e-03f*absmax*sign; // 1001
            else
                return 0.00000000f*absmax*sign; // 1000
}

__device__ unsigned char dQuantizeFP4(float x)
{
    /**
     * @brief 주어진 부동 소수점 값을 FP4 양자화된 값으로 변환합니다.
     *
     * dQuantizeFP4는 입력 부동 소수점 값을 4비트 FP4 형식으로 양자화합니다.
     * 입력 값의 부호를 결정하고, 지수와 가수를 기준으로 적절한 FP4 양자화된 값을 반환합니다.
     *
     * @param x 양자화할 부동 소수점 값입니다.
     * @return unsigned char FP4 양자화된 값입니다.
     */


    // FP4 형식의 바이어스 값 3을 사용하는 부동 소수점 양자화
    // 비교 숫자들은 FP4 형식에서 사용할 수 있는 정규화된 값을 기준으로 합니다.
    // 1. 0.29166667 = 7/24 -> 1/3과 1/4 사이의 중간 점
    // 2. 0.58333333 = 7/12 -> 1/2와 2/3 사이의 중간 점
    // 3. 0.83333333 = 5/6 -> 3/4와 1 사이의 중간 점 
    // 4. 0.41666667 = 5/12 -> 1/3과 1/2 사이의 중간 점 
    // 5. 0.0859375 = 11/128 -> 1/16과 1/8 사이의 중간 점 
    // 6. 0.20833333 = 5/24 -> 1/6과 1.4 사이의 중간 점 
    // 7. 0.00260417 = 1/384 -> 0과 1/256 사이의 중간 점

    // ex) x = 0.268456이면 가장 가까운 FP4의 값은 0b0111이 되어 반환됩니다.

    int sign = x < 0 ? 0b1000 : 0b0000; // 부호 비트 결정
    x = fabsf(x); // 부동 소수점의 절대값

    if(x > 0.29166667f) // 7/24 
        if( x > 0.583333f) // 7/12 
            if( x > 0.8333333f) // 5/6 
                return 0b0011 + sign; // 3
            else
                return 0b0010 + sign; // 2
        else
            if(x > 0.4166667f) // 5/12 
                return 0b101 + sign; // 6
            else
                return 0b100 + sign; // 4
    else 
        if(x > 0.0859375f) // 11/128 
            if(x > 0.20833333f) // 5/24 
                return 0b0111+sign; // 12
            else
                return 0b0110+sign; // 8
        else
            if(x > 0.00260417f) // 1/384 
                return 0b0001+sign; // 0.0625
            else
                return 0b0000+sign; // 0
}


template <int STOCHASTIC>
__device__ unsigned char dQuantize(float* smem_code, const float rand, float x)
{
    /**
     * @brief 입력 값을 양자화된 값으로 변환합니다.
     *
     * dQuantize는 이진 탐색 알고리즘을 사용하여 입력 값 x를 양자화된 값으로 변환합니다.
     * 확률적 양자화(stochastic quantization)를 지원하며, 이를 통해 입력 값의 분포를 기반으로 양자화가 이루어집니다.
     *
     * @tparam STOCHASTIC 확률적 양자화를 사용할지 여부를 결정합니다. 0이면 비확률적, 1이면 확률적 양자화를 사용합니다.
     * @param smem_code 양자화 코드북을 포함하는 공유 메모리 배열 포인터입니다.
     * @param rand 확률적 양자화에 사용할 난수 값입니다.
     * @param x 양자화할 입력 값입니다.
     * @return unsigned char 양자화된 값입니다.
     */

    // 이진 탐색을 위한 피벗 값 초기화
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    // 현재 탐색 중인 값의 범위 초기화 
    float lower = -1.0f;
    float upper = 1.0f;

    // 현재 탐색 중인 값 설정
    float val = smem_code[pivot];

    // 이진 탐색 수행하여 x에 가장 가까운 코드북 값을 찾음
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val) // x가 현재 값보다 큰 경우
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else // x가 현재 값보다 작거나 같은 경우
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot]; // 새로운 피벗 값 설정
    }

    // 경계값 처리
    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(!STOCHASTIC) // STOCHASTIC 양자화를 사용하지 않는 경우
    {
        if(x > val) // x가 찾은 코드북 값보다 큰 경우
        {
            float midpoint = (upper+val)*0.5f; // upper와 val의 중간값 계산
            if(x > midpoint) // x가 중간값보다 큰 경우
            {
                return upper_pivot; // upper_pivot 반환
            }
            else // x가 중간값보다 작거나 같은 경우
                return pivot; // pivot 반환
            }
        else // x가 찾은 코드북 값보다 작거나 같은 경우
        {
            float midpoint = (lower+val)*0.5f;  // lower와 val의 중간 값 계산
            if(x < midpoint) // x가 중간 값보다 작은 경우
                return lower_pivot;
            else // x가 중간 값보다 크거나 같은 경우
                return pivot;
        }
    }
    else // STOCHASTIC 양자화를 사용. 이는 동일한 입력 값이 주어지더라도 다른 양자화된 값을 반환할 수 있어 양자화 오류를 분산시키는 효과를 가짐 
    {
        if(x > val) // x가 찾은 코드북 값보다 작거나 같은 경우
        {
            float dist_to_upper = fabsf(upper-x); // x와 upper 사이의 거리 계산
            float dist_full = upper-val; // upper와 val 사이의 거리 계산
            if(rand >= dist_to_upper/dist_full) return upper_pivot; // 난수 rand가 거리 비율보다 큰 경우 upper_pivot 반환
            else return pivot;
        }
        else
        {
            float dist_to_lower = fabsf(lower-x);
            float dist_full = val-lower;
            if(rand >= dist_to_lower/dist_full) return lower_pivot;
            else return pivot;
        }
    }
}


__launch_bounds__(TH, 4) // (최대 스레드 수, 최소 병렬 블록 수)
__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n)
{
    /**
    * @brief 입력 배열 A를 4비트 값으로 양자화하고 결과를 출력 배열에 저장합니다.
    * 
    * kQuantize는 블록 단위로 입력 데이터를 처리하며, 입력 배열 A에서 값을 로드하고, 지정된 양자화 코드북을 사용하여 양자화한 다음,
    * 양자화된 값을 출력 배열 out에 저장합니다. 이 함수는 결정적(deterministic) 양자화와 확률적(stochastic) 양자화 방법을 지원합니다.
    *
    * @param code 양자화 코드북을 포함하는 float 값들의 배열 포인터로, 입력 값을 양자화된 값으로 매핑하는 데 사용됩니다.
    * @param A 양자화할 float 값들의 입력 배열 포인터입니다.
    * @param out 양자화된 값을 저장할 출력 배열 포인터입니다.
    * @param n 입력 배열 A의 요소 수입니다.
    *
    * @note 이 함수는 공유 메모리에 양자화 코드북을 저장하여 빠른 접근을 가능하게 합니다. 또한, CUB 라이브러리 함수를 이용하여
    *       효율적인 블록 단위 데이터 로드 및 저장을 수행합니다.
    *
    * @details 함수는 다음과 같은 단계로 동작합니다:
    * 1. 처리할 전체 요소 수와 블록당 유효한 항목 수를 계산합니다.
    * 2. 양자화 코드북을 공유 메모리에 로드합니다.
    * 3. 각 블록에 대해 입력 값을 로드하고, 이를 양자화한 후, 양자화된 값을 출력 배열에 저장합니다.
    * 4. dQuantize 함수를 사용하여 이진 탐색과 선택적 확률적 양자화를 수행합니다.
    *
    */
  
    // 전체 원소 수를 계산하여 필요한 블록 수를 결정
    const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
    // 각 블록에서 처리할 유효 원소 수를 계산
    int valid_items = (blockIdx.x+1 == gridDim.x) ? n - (blockIdx.x*NUM_BLOCK) : NUM_BLOCK;
    // 현재 블록의 시작 인덱스 설정
    const int base_idx = (blockIdx.x * NUM_BLOCK);

    // 입력 배열에서 로드된 값을 저장할 배열
    float vals[NUM];
    // 양자화된 값을 저장할 배열
    unsigned char qvals[NUM];

    // CUB(CUDA Unbound)는 병렬 알고리즘을 최적화하는 유틸리티. 효율적인 블록 로드와 저장을 할 수 있음
    // CUB 라이브러리의 블록 로드를 위한 타입 정의
    typedef cub::BlockLoad<float, TH, NUM, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    // CUB 라이브러리의 블록 저장을 위한 타입 정의
    typedef cub::BlockStore<unsigned char, TH, NUM, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

    // 블록 로드를 위한 CUB 라이브러리의 임시 스토리지
    __shared__ typename LoadFloat::TempStorage loadf;
    // 블록 로드를 위한 CUB 라이브러리의 임시 스토리지
    __shared__ typename StoreChar::TempStorage storec;
    // 양자화 코드북(256개의 fp)을 공유 메모리에 저장할 목적
    __shared__ float smem_code[256]; 

    // 양자화 코드북을 공유 메모리에 로드
    if(threadIdx.x < 256)
    {
        smem_code[threadIdx.x] = code[threadIdx.x];
    }


    // 각 블록에 대한 for loop
    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_BLOCK)
    {
        // 현재 블록에서 유효한 원소 수 계산
        valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

        __syncthreads();

        // 입력 배열 A에서 값을 로드해 vals 배열에 저장
        LoadFloat(loadf).Load(&(A[i]), vals, valid_items);

        // unrolling을 4개 단위로 진행. 4는 magic number는 아님
        #pragma unroll 4
        for(int j = 0; j < NUM; j++)
            // 로드된 값을 양자화하여 qvals 배열에 저장
            qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

        __syncthreads();

        // 양자화된 값을 출력 배열 out에 저장
        StoreChar(storec).Store(&(out[i]), qvals, valid_items);
    }
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE>
__global__ void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n)
{
    /**
     * @brief 블록 단위로 데이터를 FP4 형식으로 양자화합니다.
     *
     * kQuantizeBlockwise는 입력 데이터를 블록 단위로 처리하여 FP4 형식으로 양자화합니다.
     * 입력 데이터의 절대 최대값을 계산하고, 필요한 경우 확률적 양자화를 수행합니다.
     *
     * @tparam T 데이터 타입
     * @tparam BLOCK_SIZE 블록 크기
     * @tparam NUM_PER_TH 스레드당 처리할 항목 수
     * @tparam STOCHASTIC 확률적 양자화 사용 여부 (0 또는 1)
     * @tparam DATA_TYPE 데이터 형식 (FP4 또는 General8bit)
     *
     * @param code 양자화 코드
     * @param A 입력 데이터 배열
     * @param absmax 각 블록의 절대 최대값 배열
     * @param out 양자화된 출력 배열
     * @param rand 확률적 양자화를 위한 랜덤 값 배열
     * @param rand_offset 랜덤 값 오프셋
     * @param n 입력 데이터의 크기
     */


    // 그리드 내 전체 원소 수를 계산
    const int n_full = gridDim.x * BLOCK_SIZE;
    // 블록 내 유효한 원소 수 저장
    int valid_items = 0;
    // 현재 블록의 시작 인덱스를 계산
    const int base_idx = (blockIdx.x * BLOCK_SIZE);

    // 스레드당 처리할 값 배열을 선언
    T vals[NUM_PER_TH];
    // 랜덤 값을 저장할 배열
    float rand_vals[NUM_PER_TH];
    unsigned char qvals[(DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH];
    // 블록 내 최대 절대값 저장
    float local_abs_max = 0.0f;
    // 랜덤 값의 인덱스
    int local_rand_idx = 0;

    // CUB 라이브러리를 사용하여 블록 내 로드, 스토어, 리듀스 작업을 정의
    typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT; // 데이터 로드
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, (DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar; // 데이터 저장
    typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce; // 리듀스 연산
    typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat; // 랜덤 값 로드 (STOCHASTIC에 사용)

    // 공유 메모리 선언
    __shared__ typename LoadT::TempStorage loadt; // 데이터 로드 임시 스토리지
    __shared__ typename LoadFloat::TempStorage loadf; // 랜덤 값 로드 임시 스토리지
    __shared__ typename StoreChar::TempStorage storec; // 데이터 저장 임시 스토리지
    __shared__ typename BlockReduce::TempStorage reduce; // 리듀스 임시 스토리지
    __shared__ float smem_code[256]; // 양자화 코드를 저장 공유 메모리
    __shared__ float smem_absmax_value[1]; // 블록 내 최대 절대값을 저장할 공유 메모리

    if(DATA_TYPE == General8bit)
        for(int i = threadIdx.x; i < 256; i+=blockDim.x)
            smem_code[i] = code[i];

    // 블록 내 데이터를 처리합니다.
    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
    {
        // 유효 원소 수 계산
        valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
        // 초기 최대 절대값 설정
        local_abs_max = -FLT_MAX;

        __syncthreads();
        // 입력 데이터 로드
        LoadT(loadt).Load(&(A[i]), vals, valid_items, (T)0.0f);

        // 1. compute local max
        // 2. broadcast local max
        // 3. normalize inputs and quantize

        // 블록 내 절대 최대값 계산
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
            local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

        // BlockReduce로 최대값 계산
        local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max(), valid_items);
        if(threadIdx.x == 0) // 첫번째 스레드에서만 공유 메모리에 최대값 저장
            smem_absmax_value[0] = local_abs_max;

        // 최대값을 공유 메모리에 저장하고, 이를 다른 스레드로 broadcast
        __syncthreads();

        // 첫번째 스레드에서만 
        if(threadIdx.x == 0)
            absmax[i/BLOCK_SIZE] = local_abs_max; // 최대값을 absmax 배열에 저장
        else
            local_abs_max = smem_absmax_value[0]; // 최대값을 로드

        // warp 내 스레드 동기화하여 공유 메모리를 읽는 것에 대한 일관성 보장
        __syncwarp();

        local_abs_max = 1.0f/local_abs_max; // 최대값의 역수 계산 (정규화에 사용)

        // STOCHASTIC을 사용하는 경우
        if(STOCHASTIC)
        {
            // 랜덤 인덱스 계산
            local_rand_idx = ((blockIdx.x*NUM_BLOCK) + (threadIdx.x*NUM) + rand_offset) % (1024-4);
            // 랜덤 값 로드
            LoadFloat(loadf).Load(&rand[local_rand_idx], rand_vals, BLOCK_SIZE, 0);
        }

        // FP4 값을 저장할 변수
        unsigned char packed_4bit = 0;
        switch(DATA_TYPE)
        {
            case General8bit:
                #pragma unroll NUM_PER_TH
                for(int j = 0; j < NUM_PER_TH; j++)
                {
                    if(!STOCHASTIC)
                        qvals[j] = dQuantize<0>(smem_code, 0.0f, ((float)vals[j])*local_abs_max); // No stochastic
                    else
                        qvals[j] = dQuantize<1>(smem_code, rand_vals[j], ((float)vals[j])*local_abs_max);
                }
                break;
            case FP4:
                #pragma unroll NUM_PER_TH
                for(int j = 0; j < NUM_PER_TH/2; j++)
                {
                    packed_4bit |= dQuantizeFP4(((float)vals[2*j])*local_abs_max) << 4; // 상위 4비트 저장(ex. 00001111 << 4 == 11110000)
                    packed_4bit |= dQuantizeFP4(((float)vals[2*j+1])*local_abs_max); // 하위 4비트 저장
                    qvals[j] = packed_4bit; // fp4 값 저장
                }
                break;
        }

        __syncthreads();
        // 양자화된 값을 출력 배열에 저장
        StoreChar(storec).Store(&(out[(DATA_TYPE > 0) ? i/2 : i]), qvals, (DATA_TYPE > 0) ? (valid_items+1)/2 : valid_items);
    }
}

template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n)
{
    /**
     * @brief 블록 단위로 FP4 또는 8비트 형식의 데이터를 디양자화하는 커널 함수.
     *
     * kDequantizeBlockwise는 입력된 양자화된 데이터를 블록 단위로 디양자화합니다. 
     * FP4 형식의 데이터를 디양자화하여 원래의 부동 소수점 값으로 복원합니다.
     *
     * @tparam T 데이터 타입
     * @tparam TILE_SIZE 타일 크기
     * @tparam THREADS 블록 당 스레드 수
     * @tparam NUM_PER_TH 스레드당 처리할 항목 수
     * @tparam DATA_TYPE 데이터 형식 
     *
     * @param code 디양자화 코드
     * @param A 양자화된 입력 데이터 배열
     * @param absmax 각 블록의 절대 최대값 배열
     * @param out 디양자화된 출력 배열
     * @param blocksize 블록 크기
     * @param n 입력 데이터의 크기
     */

    // 그리드 내 전체 원소 수 계산
    const int n_load = (gridDim.x * TILE_SIZE);
    // 로드할 요효 원소 수를 저장
    int valid_items_load = 0;
    // 저장할 유요 원소 수를 저장
    int valid_items_store = 0;
    // 현재 블록의 시작 인덱스
    const int base_idx = (blockIdx.x * TILE_SIZE);

    // 스레드 당 처리할 값 배열 선언
    T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
    // 양자화된 값을 저장할 배열
    unsigned char qvals[NUM_PER_TH];
    // 블록 내 최대 절대값 저장
    float local_abs_max = -FLT_MAX;

    // CUB 라이브러리를 사용하여 블록 내 로드 및 스토어 작업 정의
    typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar; // 데이터 로드 정의
    typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT; // 데이터 저장 정의

    // 데이터 로드/저장의 임시 스토리지를 smem에 할당
    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    // tile 단위로 데이터 처리
    for (unsigned int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE)
    {
        if(DATA_TYPE > 0)
        {
            valid_items_load = (n+1)/2 - i > TILE_SIZE ? TILE_SIZE : (n+1)/2 - i;
            valid_items_store = n - i*2 > TILE_SIZE*2 ? TILE_SIZE*2 : n - i*2;
        }
        else
        {
            valid_items_load = n - i > TILE_SIZE ? TILE_SIZE : n - i; // 로드할 유효 원소 수를 계산
            valid_items_store = n - i > TILE_SIZE ? TILE_SIZE : n - i; // 저장할 유효 원소 수를 계산
        }
        // 절대 최대값을 로드
        local_abs_max = __ldg(&absmax[(i+threadIdx.x*NUM_PER_TH)/(blocksize)]);

        __syncthreads();
        // 양자화된 입력 데이터를 로드
        LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

        switch(DATA_TYPE)
        {
            case General8bit:
                #pragma unroll NUM_PER_TH
                for(int j = 0; j < NUM_PER_TH; j++)
                vals[j] = __ldg(&code[qvals[j]])*local_abs_max;
                break;
            case FP4:
                #pragma unroll NUM_PER_TH
                for(int j = 0; j < NUM_PER_TH; j++)
                {
                    vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max); // 상위 4비트를 dequantize
                    vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max); // 하위 4비트를 dequantize
                }
                break;
        }

        __syncthreads();
        // dequantize된 값을 출력 배열에 저장
        StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
    }
}

__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n)
{
    /**
     * @brief 양자화된 데이터를 디양자화하는 CUDA 커널 함수.
     *
     * kDequantize는 양자화된 입력 데이터 배열을 dequantize하여 원래의 부동 소수점 값으로 복원합니다.
     * 각 스레드는 고유한 인덱스를 계산하고, 공유 메모리에서 양자화 코드 배열을 로드하여 dequnatize 작업을 수행합니다.
     *
     * @param code dequantized된 코드 배열
     * @param A 양자화된 입력 데이터 배열
     * @param out dequnatized된 출력 데이터 배열
     * @param n 입력 데이터의 크기
     */

    // 그리드 내 모든 스레드의 총 개수 계산
    const unsigned int numThreads = blockDim.x * gridDim.x;
    // 현재 스레드의 전역 인덱스
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // 256 크기의 공유 메모리 배열 선언
    __shared__ float smem_code[256];

    // 첫 번째 256개의 스레드는 dequantize 코드를 공유 메모리에 로드
	if(threadIdx.x < 256)
	{
		smem_code[threadIdx.x] = code[threadIdx.x];
	}

    // 모든 스레드가 dequnatize된 코드를 로드할 때까지 동기화
	__syncthreads();

    // 전체 입력 데이터 배열을 처리하는 for loop
	for (int i = idx;i < n; i += numThreads)
	{
        // 양자화된 값 A[i]를 사용하여 dequantize된 값을 공유 메모리에서 가져와 출력 배열에 저장
		out[i] = smem_code[A[i]];
	}
}


#define WARPS 3
template <typename T, int BITS, int THREADS> __global__ void gemm_device(int M, int N, int K, T * __restrict__ const A,  T* B,  T * out,  int lda, int ldb, int ldc)
{
    /**
     * @brief General Matrix-Matrix Multiplication (GEMM) CUDA 커널 함수.
     *
     * gemm_device는 행렬 A와 B의 곱을 계산하여 행렬 out에 저장합니다. NVIDIA의 Warp Matrix Multiply Accumulate (WMMA) API를 사용하여 
     * FP16 데이터 타입의 행렬 곱셈을 최적화합니다.
     *
     * @tparam T 데이터 타입 (예: half)
     * @tparam BITS 비트 수
     * @tparam THREADS 블록 당 스레드 수
     *
     * @param M 행렬 A의 행 수 및 행렬 C의 행 수
     * @param N 행렬 B의 열 수 및 행렬 C의 열 수
     * @param K 행렬 A의 열 수 및 행렬 B의 행 수
     * @param A 행렬 A에 대한 포인터
     * @param B 행렬 B에 대한 포인터
     * @param out 행렬 C에 대한 포인터
     * @param lda 행렬 A의 leading dimension
     * @param ldb 행렬 B의 leading dimension
     * @param ldc 행렬 C의 leading dimension
     */

    // CUDA 아키텍쳐 버전이 7.5 이상인지 확인
    #if __CUDA_ARCH__ >= 750
    // WMMA 네임스페이스 사용
    using namespace nvcuda;

    int col_offset = blockIdx.x *32; // 블록의 column offset
    const int warp_id = threadIdx.x / 32; // 현재 스레드의 warp id
    const int half_warp_id = threadIdx.x / 16; // 현재 스레드의 half warp id -> half warp를 사용하는 이유는 8x16과 16x32 프래그먼트를 16개 스레드로 처리하기 위함
    const int half_warp_lane = threadIdx.x % 16; // 현재 스레드의 half 내 위치
    const int batch_size_warps = (WARPS-1)*2; // 배치 사이즈 warp 수를 계산
    const int val_per_iter = blockDim.x-32; // iter 당 처리할 값의 수

    T local_A[4]; // 로컬 변수로 A의 값을 저장 
    T local_B[128]; // 로컬 변수로 B의 값을 저장

    const int a_tile_offset = 16; // A 타일의 offset
    const int b_tile_offset = (16*32 + 16); // B 타일의 offset

    __shared__ T smem_A[8*16 + (2*16*(batch_size_warps-1))]; // 공유 메모리에 A 타일을 저장
    __shared__ T smem_B[2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))]; // 공유 메모리에 B 타일을 저장

    /**
    * WMMA(Warp Matrix Multiply Accumulate)는 행렬 곱 및 누산 작업을 효율적으로 하는 API 
    * WMMA fragment는 Tensor core에서 행렬 연산을 효율적으로 수행하기 위한 데이터 구조로, 행렬의 부분 집합을 나타냄
    */

    // WMMA 프래그먼트 정의
    // A: 8x16, B: 16x32, C: 8x32, 데이터 타입: half
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> c_frag; // 누산기. 행렬 곱의 결과를 저장
    wmma::fill_fragment(c_frag, 0.0f); // 누산기 프래그먼트를 0으로 초기화

    int ticktock = 0; // 반복을 위한 변수
    int idx = 0 + threadIdx.x; // 현재 스레드의 인덱스
    int loaded_values = 0; // 로드된 값의 수

    // 데이터 미리 가져오기(prefetch)
    if(idx < K && warp_id < (WARPS-1)) // 현재 스레드의 인덱스가 K보다 작고, warp id가 WARPS-1보다 작은 경우
    {
        // 현재 스레드가 처음으로 데이터를 로드할 때
        if(loaded_values == 0)
        {
            // A 행렬의 값 4개 로드
            local_A[0] = A[idx]; // A 행렬의 첫번째 값 로드
            local_A[1] = A[idx+(1*val_per_iter)]; // A 행렬의 두번째 값 로드
            local_A[2] = A[idx+(2*val_per_iter)]; // A 행렬의 세번째 값 로드
            local_A[3] = A[idx+(3*val_per_iter)]; // A 행렬의 네번째 값 로드

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
            {
                // B 행렬의 값 128개  로드
                local_B[col] = B[(col_offset+col)*ldb+idx];
                local_B[col+32] = B[(col_offset+col)*ldb+idx+(1*val_per_iter)];
                local_B[col+64] = B[(col_offset+col)*ldb+idx+(2*val_per_iter)];
                local_B[col+96] = B[(col_offset+col)*ldb+idx+(3*val_per_iter)];
            }
            loaded_values = 3;
        }
        // 로드된 값이 0이 아닌 경우
        else
        {
            if(loaded_values == 3)
            {
                local_A[0] = local_A[1]; // 이전에 로드한 값을 이동
                #pragma unroll 32
                for(int col = 0; col < 32; col++)
                    local_B[col] = local_B[col+(32)]; // 다음 32개의 값을 로드
            }
            else if(loaded_values == 2)
            {
                local_A[0] = local_A[2];
                #pragma unroll 32
                for(int col = 0; col < 32; col++)
                    local_B[col] = local_B[col+(64)]; // 다음 64개의 값을 로드
            }
            else
            {
                local_A[0] = local_A[3];
                #pragma unroll 32
                for(int col = 0; col < 32; col++)
                    local_B[col] = local_B[col+(96)]; // 다음 96개의 값을 로드
            }
            loaded_values--;
        }
        // 공유 메모리에 A 타일을 저장
        smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

        // 공유 메모리에 B 타일을 저장
        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
    }
    // warp id가 WARPS-1보다 작은 경우
    else if(warp_id < (WARPS-1))
    {
        local_A[0] = T(0.0); // 초기화
        smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            local_B[col] = 0.0f;

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
    }
    ticktock = ticktock == 0 ? 1 : 0; // ticktock을 토글하여 공유 메모리에서 A와 B 타일의 현재 위치를 번갈아 가며 사용

    // 반복해서 남은 데이터를 로드하고 연산
    for(int base_idx = blockDim.x-32; base_idx < K; base_idx+=blockDim.x-32)
    {
        idx = base_idx + threadIdx.x;

        __syncthreads();
        if(idx < K && warp_id < (WARPS-1))
        {
            if(loaded_values == 0)
            {
                local_A[0] = A[idx];
                local_A[1] = A[idx+(1*val_per_iter)];
                local_A[2] = A[idx+(2*val_per_iter)];
                local_A[3] = A[idx+(3*val_per_iter)];

                #pragma unroll 32
                for(int col = 0; col < 32; col++)
                {
                    local_B[col] = B[(col_offset+col)*ldb+idx];
                    local_B[col+32] = B[(col_offset+col)*ldb+idx+(1*val_per_iter)];
                    local_B[col+64] = B[(col_offset+col)*ldb+idx+(2*val_per_iter)];
                    local_B[col+96] = B[(col_offset+col)*ldb+idx+(3*val_per_iter)];
                }
                loaded_values = 3;

            }
            else
            {
                if(loaded_values == 3)
                {
                    local_A[0] = local_A[1];
                    #pragma unroll 32
                    for(int col = 0; col < 32; col++)
                        local_B[col] = local_B[col+(32)];
                }
                else if(loaded_values == 2)
                {
                    local_A[0] = local_A[2];
                    #pragma unroll 32
                    for(int col = 0; col < 32; col++)
                        local_B[col] = local_B[col+(64)];
                }
                else
                {
                    local_A[0] = local_A[3];
                    #pragma unroll 32
                    for(int col = 0; col < 32; col++)
                        local_B[col] = local_B[col+(96)];
                }
                loaded_values--;
            }

            smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
        }
        else if(warp_id < (WARPS-1))
        {
            local_A[0] = T(0.0);
            smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                local_B[col] = 0.0f;

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
        }
        ticktock = ticktock == 0 ? 1 : 0;

        if(warp_id == (WARPS-1))
            for(int k = 0; k < batch_size_warps; k++)
            {
                // (smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]) 메모리 주소에서 데이터를 로드하여 프레그먼트에 저장
                wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); // a_frag 로드. 111 mu
                wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // b_frag 로드. 35 mu
                // a_frag와 b_frag의 행렬 곱을 계산한 후 그 결과를 c_frag에 저장
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
    }

    __syncthreads();
    if(warp_id != (WARPS-1)){ return; }
    int warp_lane = threadIdx.x % 32;

    ticktock = ticktock == 0 ? 1 : 0;
    for(int k = 0; k < batch_size_warps; k++)
    {
        wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
        wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 129 mu
    if(warp_id == (WARPS-1))
        // (smem_A[0]) 메모리 주소에 c_frag의 데이터를 저장
        wmma::store_matrix_sync(&(smem_A[0]), c_frag, 32, wmma::mem_row_major); // row_major(행 우선) 레이아웃으로 저장

    if(col_offset + warp_lane < M)
        out[col_offset + warp_lane] = smem_A[warp_lane]; // 결과를 out에 저장
    #endif
}


__device__ static float q_data[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};
template <typename T, int THREADS> __global__ void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize)
{
    /**
     * @brief 4비트 양자화 행렬 곱셈 CUDA 커널 함수.
     *
     * kgemm_4bit_inference는 4비트 양자화된 행렬 B를 사용하여 행렬 A와 곱셈을 수행하고, 그 결과를 행렬 out에 저장합니다. 
     * NVIDIA의 Warp Matrix Multiply Accumulate (WMMA) API를 사용하여 FP16 데이터 타입의 행렬 곱셈을 최적화합니다.
     *
     * @tparam T 데이터 타입 (예: half)
     * @tparam THREADS 블록 당 스레드 수
     *
     * @param M 행렬 A의 행 수 및 행렬 C의 행 수
     * @param N 행렬 B의 열 수 및 행렬 C의 열 수
     * @param K 행렬 A의 열 수 및 행렬 B의 행 수
     * @param A 행렬 A에 대한 포인터
     * @param B 4비트 양자화된 행렬 B에 대한 포인터
     * @param absmax B의 절대 최대값 배열에 대한 포인터
     * @param out 행렬 C에 대한 포인터
     * @param lda 행렬 A의 leading dimension
     * @param ldb 행렬 B의 leading dimension
     * @param ldc 행렬 C의 leading dimension
     * @param blocksize 블록 크기
     */

    // CUDA 아키텍쳐 버전이 7.5 이상인지 확인
    #if __CUDA_ARCH__ >= 750
    // WMMA 네임스페이스 사용
    using namespace nvcuda;

    int col_offset = blockIdx.x *32; // 블록의 column offset
    const int warp_id = threadIdx.x / 32; // 현재 스레드의 warp id
    const int warp_idx = threadIdx.x % 32; // 현재 스레드의 warp 내 위치
    const int half_warp_id = threadIdx.x / 16; // 현재 스레드의 half warp id
    const int half_warp_lane = threadIdx.x % 16; // 현재 스레드의 half warp 내 위치
    const int batch_size_warps = (WARPS-1)*2; // 배치 사이즈 warp 수를 계산

    T quant_map[16]; // 4비트 양자화 맵

    #pragma unroll 16
    for(int i = 0; i < 16; i++)
        quant_map[i] = q_data[i]; // 양자화 맵 초기화

    T local_A[2]; // 로컬 변수로 A의 값을 저장
    T local_B[64]; // 로컬 변수로 B의 값을 저장
    unsigned char local_B_4bit[32]; // 로컬 변수로 4비트 B의 값을 저장


    const int a_tile_offset = 16; // A 타일의 offset
    const int b_tile_offset = (16*32 + 16); // B 타일의 offset

    __shared__ T smem_A[8*16 + (16*(batch_size_warps-1))]; // 공유 메모리에 A 타일을 저장
    __shared__ T smem_B[2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))]; // 공유 메모리에 B 타일을 저장
    __shared__ T smem_C[8*32]; // 공유 메모리에 C 타일을 저장

    // WMMA 프래그먼트 정의
    // A: 8x32, B: 32x16, C: 8x32, 데이터 타입: half
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> c_frag; // 누산기. 행렬 곱의 결과를 저장
    wmma::fill_fragment(c_frag, 0.0f); // 누산기 프래그먼트를 0으로 초기화

    // C 타일 초기화
    for(int i = threadIdx.x; i < (8*32); i+=blockDim.x)
        smem_C[i] = 0.0f;

    __syncthreads(); // 모든 스레드가 이 지점에 도달할 때까지 대기

    int ticktock = 0; // 반복을 위한 변수
    int idx = 0 + threadIdx.x; // 현재 스레드의 인덱스
    int loaded_values = 0; // 로드된 값의 수

    // 데이터 미리 가져오기(prefetch)
    if(idx < K && warp_id < (WARPS-1)) // 현재 스레드의 인덱스가 K보다 작고, warp id가 WARPS-1보다 작은 경우
    {
        if(loaded_values == 0) // 처음으로 데이터를 로드할 때
        {
            local_A[0] = A[idx]; // A 행렬의 첫번째 값 로드
            local_A[1] = A[idx+blockDim.x-32]; // A 행렬의 두번째 값 로드

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                local_B_4bit[col] = B[(col_offset+col)*ldb+idx]; // B 행렬의 값 로드

            loaded_values = 1; // 로드된 값의 수 업데이트
        }
        else
        {
            local_A[0] = local_A[1]; // 이전에 로드한 값을 이동
            loaded_values--;

            #pragma unroll 64
            for(int col = 0; col < 64; col+=2)
            {
                local_B[col] = quant_map[160*(local_B_4bit[col/2] >> 4)+warp_idx]*T(17.0); // B 행렬의 첫번째 값 로드 및 변환
                local_B[col+1] = quant_map[160*(local_B_4bit[col/2] & 0x0F)+warp_idx]*T(17.0); // B 행렬의 두번째 값 로드 및 변환
            }
        }

        // 공유 메모리에 A 타일 저장
        smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

        // 공유 메모리에 B 타일 저장
        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
    }
    else if(warp_id < (WARPS-1)) // warp id가 WARPS-1보다 작은 경우
    {
        local_A[0] = T(0.0); // 로컬 변수 초기화
        smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f; // 공유 메모리에 초기화된 값 저장

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            local_B[col] = 0.0f;

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
            smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f; // 공유 메모리에 초기화된 값 저장
    }
    ticktock = ticktock == 0 ? 1 : 0;

    // 반복해서 남은 데이터를 로드하고 연산
    for(int base_idx = blockDim.x-32; base_idx < K; base_idx+=blockDim.x-32)
    {
        idx = base_idx + threadIdx.x; // 현재 스레드의 인덱스 업데이트
        if(idx < K && warp_id < (WARPS-1))
        {
            if(loaded_values == 0)
            {
                local_A[0] = A[idx];
                local_A[1] = A[idx+blockDim.x-32];

                #pragma unroll 32
                for(int col = 0; col < 32; col++)
                {
                    local_B_4bit[col] = B[(col_offset+col)*ldb+idx];
                    local_B_4bit[col+16] = B[(col_offset+col)*ldb+idx];
                }

                loaded_values = 1;
            }
            else
            {
                local_A[0] = local_A[1];
                loaded_values--;

                int absidx = (idx + col_offset)/blocksize;
                half local_absmax = __ldg(&(absmax[absidx]));

                #pragma unroll 64
                for(int col = 0; col < 64; col+=2)
                {
                    local_B[col] = quant_map[(local_B_4bit[col/2] >> 4)]*T(absidx);
                    local_B[col+1] = quant_map[(local_B_4bit[col/2] & 0x0F)]*T(absidx);
                }
            }

            smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
        }
        else if(warp_id < (WARPS-1))
        {
            local_A[0] = T(0.0);
            smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                local_B[col] = 0.0f;

            #pragma unroll 32
            for(int col = 0; col < 32; col++)
                smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
        }
        ticktock = ticktock == 0 ? 1 : 0;

        if(warp_id == (WARPS-1))
            for(int k = 0; k < batch_size_warps; k++)
            {
                wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
                wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
    }

    __syncthreads();
    if(warp_id != (WARPS-1)){ return; }
    int warp_lane = threadIdx.x % 32;

    ticktock = ticktock == 0 ? 1 : 0;
    for(int k = 0; k < batch_size_warps; k++)
    {
        // a_frag, b_frag 로드
        wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
        wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // 행렬 곱 연산 수행
    }

    // 129 mu
    if(warp_id == (WARPS-1))
        wmma::store_matrix_sync(&(smem_C[0]), c_frag, 32, wmma::mem_row_major); // c_frag 저장

    if(col_offset + warp_lane < M)
        out[col_offset + warp_lane] = smem_C[warp_lane]; // 결과를 out 배열에 저장
    #endif
}

#define num_values_4bit 32
template <typename T, int THREADS, int BITS> __global__ void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{
    /**
     * @brief 4비트 양자화 행렬 곱셈 CUDA 커널 함수.
     *
     * kgemm_4bit_inference_naive는 4비트 양자화된 행렬 B를 사용하여 행렬 A와 곱셈을 수행하고, 
     * 그 결과를 행렬 out에 저장합니다. 성능 최적화를 위해 CUDA의 Warp Reduce와 공유 메모리를 사용합니다.
     *
     * @tparam T 데이터 타입 (예: half, float)
     * @tparam THREADS 블록 당 스레드 수
     * @tparam BITS 데이터 비트 수 (예: 4, 8, 16)
     *
     * @param M 행렬 A의 행 수 및 행렬 C의 행 수
     * @param N 행렬 B의 열 수 및 행렬 C의 열 수
     * @param K 행렬 A의 열 수 및 행렬 B의 행 수
     * @param A 행렬 A에 대한 포인터
     * @param B 4비트 양자화된 행렬 B에 대한 포인터
     * @param absmax B의 절대 최대값 배열에 대한 포인터
     * @param datatype 양자화 데이터 타입 배열에 대한 포인터
     * @param out 행렬 C에 대한 포인터
     * @param lda 행렬 A의 leading dimension
     * @param ldb 행렬 B의 leading dimension
     * @param ldc 행렬 C의 leading dimension
     * @param blocksize 블록 크기
     */


    // 각 thread block 당:
    // [32, warps] 크기의 청크 단위로 단계적으로 로드: 1x32 * [32, warps] -> [1, warps]
    // 4 warps -> 반복 당 4개의 로드
    // 1x32 * 32x4 -> thread block 당 1x4 출력

    /**
     * Warp Reduce는 warp 내에서 병렬적으로  데이터를 reduce하는 API
     * 
     * 보통 cub::WarpReduce<T>으로 템플릿 인스턴스화 하여 사용
     * __shared__ typename WarpReduce::TempStorage temp_storage[32]; 이런 방식으로 임시 저장 공간을 만들어서 각 warp의 스레드 간 데이터를 공유할 때 사용
     * 
     * Reduce 함수 : Sum, Reduce, Min, Max가 있음. Reduce는 지정된 연산자를 사용하여 reduce함
    */

    typedef cub::WarpReduce<float> WarpReduce; // WarpReduce 타입 정의
    __shared__ typename WarpReduce::TempStorage temp_storage[THREADS/32]; // 공유 메모리 스토리지 정의

    const int warp_idx = threadIdx.x / 32; // 현재 스레드의 warp 인덱스
    const int warp_lane = threadIdx.x % 32; // 현재 스레드의 warp 내 위치
    const int row_B = (THREADS/32)*blockIdx.x + warp_idx; // B 행렬의 행 인덱스 계산
    const int num_values_8bit = num_values_4bit/2; // 8비트 값의 수 계산
    float local_C = 0.0f; // 로컬 변수로 결과 값 초기화

    unsigned char local_B_4bit[num_values_8bit]; // 4비트 B 값을 저장할 로컬 배열
    T local_B[num_values_4bit/4]; // 변환된 B 값을 저장할 로컬 배열
    T local_A[num_values_4bit/4]; // A 값을 저장할 로컬 배열
    __shared__ T quant_map[16]; // 양자화 맵을 저장할 공유 메모리
    T local_absmax = T(0.0f); // 로컬 변수로 absmax 값 초기화

    // 양자화 맵 초기화
    for(int i = threadIdx.x; i < 16; i++)
        quant_map[i] = T(datatype[i]);
    __syncthreads();

    // A: [1, K]
    // B: [N, K]
    // 내부 인덱스 루프
    for(int inner_idx = warp_lane*num_values_4bit; inner_idx < K; inner_idx += 32*num_values_4bit)
    {
        int inner_idx_halved = inner_idx/2; // 내부 인덱스를 절반으로 나눔
        int offset_B = ldb*row_B; // B 행렬의 오프셋 계산
        int absidx = ((2*offset_B)+inner_idx)/blocksize; // 절대 인덱스 계산
        local_absmax = __ldg(&(absmax[absidx])); // absmax 값 로드

        // B 행렬의 행 인덱스가 M보다 작은 경우
        if(row_B < M)
        {
            // 내부 인덱스가 범위 내에 있는 경우
            if((inner_idx_halved + num_values_8bit) < (K/2))
            {
                // 성능 최적화를 위해 중요한 부분
                reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] = reinterpret_cast<int4*>(B)[(offset_B+(inner_idx_halved))/(num_values_8bit)];
            }
            // 내부 인덱스가 범위를 벗어나는 경우
            else
            {
                #pragma unroll
                for(int j = 0; j < (num_values_8bit); j++) // 8비트 값을 반복하여 로드
                    if((inner_idx_halved) + j < (K/2))
                        local_B_4bit[j] = B[offset_B+inner_idx_halved + j]; // B 값 로드
                    else
                        local_B_4bit[j] = 0b01110111; // 범위를 벗어난 경우 기본값 설정
            }
        }
        // B 행렬의 행 인덱스가 M보다 크거나 같은 경우
        else
        {
            // 기본값 설정
            #pragma unroll
            for(int j = 0; j < (num_values_8bit); j++)
                local_B_4bit[j] = 0b01110111;
        }

        for(int i = 0; i < 4; i++)
        {
            // 4비트 값을 변환하여 로컬 배열에 저장
            #pragma unroll
            for(int k = 0; k < num_values_8bit/4; k++)
            {
                #if __CUDA_ARCH__ >= 800
                local_B[k*2] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*local_absmax;
                local_B[k*2 + 1] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*local_absmax;
                #else
                // bf16 곱셈을 지원하지 않는 경우
                local_B[k*2] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*(float)local_absmax);
                local_B[k*2 + 1] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*(float)local_absmax);
                #endif
            }

            // 내부 인덱스가 K 범위 내에 있는 경우
            if(inner_idx+(num_values_4bit/4) + (i*num_values_4bit/4) < K)
            {
                // 성능 최적화를 위해 중요한 부분
                if(BITS==16) // 16비트인 경우
                {
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/4) + i];
                }
                else // 16비트가 아닌 경우
                {
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 0];
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 1];
                }
            }
            // 내부 인덱스가 K 범위를 벗어나는 경우
            else
                #pragma unroll
                for(int k = 0; k < num_values_4bit/4; k++)
                    if(inner_idx + (i*num_values_4bit/4) + k < K)
                        local_A[k] = A[inner_idx + k + (i*num_values_4bit/4)];
                    else
                        local_A[k] = T(0.0f); // 범위를 벗어난 경우 기본값 설정

            // 결과를 float로 누적; Ampere의 경우 성능 저하가 있을 수 있지만, 출력의 오차가 적음
            #pragma unroll
            for(int k = 0; k < num_values_4bit/4; k++) // 누산 연산 수행
            {
                #if __CUDA_ARCH__ >= 800
                local_C += (float)(local_A[k]*local_B[k]);
                #else
                // bf16 곱셈을 지원하지 않는 경우
                local_C += ((float)local_A[k]*(float)local_B[k]);
                #endif
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C); // 각 warp 내에서 결과를 합산

    if(row_B < M && warp_lane == 0) // B 행렬의 행 인덱스가 M보다 작고, warp 내 위치가 0인 경우
        out[row_B] = T(local_C); // 결과를 out 배열에 저장
}


// template 인스턴스
// 템플릿 인스턴스를 통해 컴파일러가 템플릿 함수를 미리 생성하도록 강제하여 전체 프로젝트의 컴파일 시간 단축
// 그리고 중복된 템플릿 인스턴스 생성 방지하여 바이너리 크기 최적화


// these are not used and make no sense, but the compiler needs them
//template __global__ void gemm_device<float, 16, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 256>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 192>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 160>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 128>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 32>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 64>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 32, 96>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
// these are not used and make no sense, but the compiler needs them

//template __global__ void gemm_device<float, 32, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 256>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 192>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 160>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 128>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
//template __global__ void gemm_device<float, 32, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 32>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 64>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);
template __global__ void gemm_device<half, 16, 96>(int M, int N, int K, half * __restrict__ const A,  half* B,  half * out,  int lda, int ldb, int ldc);

template __global__ void kgemm_4bit_inference<half, 96>(int M, int N, int K, half * __restrict__ const A, unsigned char *B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template __global__ void kgemm_4bit_inference<half, 128>(int M, int N, int K, half * __restrict__ const A, unsigned char *B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template __global__ void kgemm_4bit_inference<half, 160>(int M, int N, int K, half * __restrict__ const A, unsigned char *B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template __global__ void kgemm_4bit_inference<half, 256>(int M, int N, int K, half * __restrict__ const A, unsigned char *B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);

template __global__ void kgemm_4bit_inference_naive<half, 128, 16>(int M, int N, int K, half * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize);
template __global__ void kgemm_4bit_inference_naive<__nv_bfloat16, 128, 16>(int M, int N, int K, __nv_bfloat16 * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize);
template __global__ void kgemm_4bit_inference_naive<float, 128, 32>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

template __device__ unsigned char dQuantize<0>(float* smem_code, const float rand, float x);
template __device__ unsigned char dQuantize<1>(float* smem_code, const float rand, float x);


#define MAKE_kQuantizeBlockwise(dtype, blocksize, num_per_thread, stochastic, data_type_name) \
template __global__ void kQuantizeBlockwise<dtype, blocksize, num_per_thread, stochastic, data_type_name>(float * code, dtype * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n); \

MAKE_kQuantizeBlockwise(half,  4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half,  4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(half,  2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half,  1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half,   512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half,   256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half,   128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half,    64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half,  4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(half,  2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(half,  1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(half,   512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(half,   256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(half,   128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(half,    64, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(float, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,   64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float, 2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float, 1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float,  512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,  256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,  128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,   64, 2, 0, FP4)

MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, 0, FP4)

template __global__ void kDequantizeBlockwise<half, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, half *out, const int blocksize, const int n);
template __global__ void kDequantizeBlockwise<half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, half *out, const int blocksize, const int n);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, __nv_bfloat16 *out, const int blocksize, const int n);
template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, __nv_bfloat16 *out, const int blocksize, const int n);
