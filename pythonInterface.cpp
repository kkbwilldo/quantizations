#include <Python.h>
#include <cuda_fp16.h>
#include "ops.cuh"

/*
Pybind11을 사용할 경우, ctypes 데이터 형을 C++에게 인자를 넘겨주는 상황에서 데이터타입 불일치가 발생함.
이 문제를 해결하지 못해서 CMakeLists.txt 형식으로 빌드하기로 결정...ㅠㅠ
*/


// ##################################################################################################################################
// ops.cu의 템플릿 인스턴스를 사용하는 명시적 함수 정의

// gemm_4bit
void gemm_4bit_inference_naive_fp16(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive<half, 16>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }
void gemm_4bit_inference_naive_bf16(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive<__nv_bfloat16, 16>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }
void gemm_4bit_inference_naive_fp32(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive<float, 32>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

// quantize_4bit
void quantizeBlockwise_fp32_fp4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<float, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp16_fp4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<half, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_bf16_fp4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<__nv_bfloat16, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }

// dequantize_4bit
void dequantizeBlockwise_fp32_fp4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float, FP4>(NULL, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_fp16_fp4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half, FP4>(NULL, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_bf16_fp4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise<__nv_bfloat16, FP4>(NULL, A, absmax, out, blocksize, n); }

// quantize_blockwise
void quantizeBlockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<float, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<half, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_bf16(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<__nv_bfloat16, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }

// dequantize_blockwise
void dequantizeBlockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half, General8bit>(code, A, absmax, out, blocksize, n); } 
void dequantizeBlockwise_bf16(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise<__nv_bfloat16, General8bit>(code, A, absmax, out, blocksize, n); }

// ##################################################################################################################################
// 명시적 함수 정의. c를 앞에 붙여 구분

// gemm_4bit
// 명시적 함수 정의
void cgemm_4bit_inference_naive_fp32(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive_fp32(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }
void cgemm_4bit_inference_naive_fp16(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive_fp16(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }
void cgemm_4bit_inference_naive_bf16(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize){ gemm_4bit_inference_naive_bf16(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

// quantize_4bit
void cquantize_blockwise_fp32_fp4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n); }
void cquantize_blockwise_fp16_fp4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n); }
void cquantize_blockwise_bf16_fp4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n); }

// dequantize_4bit
void cdequantize_blockwise_fp32_fp4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n); }
void cdequantize_blockwise_fp16_fp4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n); }
void cdequantize_blockwise_bf16_fp4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n); }

// quantize_blockwise
void cquantize_blockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp32(code, A, absmax, out, blocksize, n); }
void cquantize_blockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp16(code, A, absmax, out, blocksize, n); }
void cquantize_blockwise_bf16(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_bf16(code, A, absmax, out, blocksize, n); }

// dequantize_blockwise
void cdequantize_blockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n); }
void cdequantize_blockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16(code, A, absmax, out, blocksize, n); }
void cdequantize_blockwise_bf16(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise_bf16(code, A, absmax, out, blocksize, n); }

// ##################################################################################################################################
// Python wrapper functions

// GEMM 4bit
static PyObject* py_cgemm_4bit_inference_naive_fp32(PyObject* self, PyObject* args) {
    unsigned long long A_ptr, B_ptr, absmax_ptr, datatype_ptr, out_ptr;
    int m, n, k, lda, ldb, ldc, blocksize;

    if (!PyArg_ParseTuple(args, "iiiKKKKKiiii", &m, &n, &k, &A_ptr, &B_ptr, &absmax_ptr, &datatype_ptr, &out_ptr, &lda, &ldb, &ldc, &blocksize)) {
        return NULL;
    }

    float* A = reinterpret_cast<float*>(A_ptr);
    unsigned char* B = reinterpret_cast<unsigned char*>(B_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    float* datatype = reinterpret_cast<float*>(datatype_ptr);
    float* out = reinterpret_cast<float*>(out_ptr);

    cgemm_4bit_inference_naive_fp32(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);

    Py_RETURN_NONE;
}

static PyObject* py_cgemm_4bit_inference_naive_fp16(PyObject* self, PyObject* args) {
    unsigned long long A_ptr, B_ptr, absmax_ptr, datatype_ptr, out_ptr;
    int m, n, k, lda, ldb, ldc, blocksize;

    if (!PyArg_ParseTuple(args, "iiiKKKKKiiii", &m, &n, &k, &A_ptr, &B_ptr, &absmax_ptr, &datatype_ptr, &out_ptr, &lda, &ldb, &ldc, &blocksize)) {
        return NULL;
    }

    half* A = reinterpret_cast<half*>(A_ptr);
    unsigned char* B = reinterpret_cast<unsigned char*>(B_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    float* datatype = reinterpret_cast<float*>(datatype_ptr);
    half* out = reinterpret_cast<half*>(out_ptr);

    cgemm_4bit_inference_naive_fp16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);

    Py_RETURN_NONE;
}

static PyObject* py_cgemm_4bit_inference_naive_bf16(PyObject* self, PyObject* args) {
    unsigned long long A_ptr, B_ptr, absmax_ptr, datatype_ptr, out_ptr;
    int m, n, k, lda, ldb, ldc, blocksize;

    if (!PyArg_ParseTuple(args, "iiiKKKKKiiii", &m, &n, &k, &A_ptr, &B_ptr, &absmax_ptr, &datatype_ptr, &out_ptr, &lda, &ldb, &ldc, &blocksize)) {
        return NULL;
    }

    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(A_ptr);
    unsigned char* B = reinterpret_cast<unsigned char*>(B_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    float* datatype = reinterpret_cast<float*>(datatype_ptr);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_ptr);

    cgemm_4bit_inference_naive_bf16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);

    Py_RETURN_NONE;
}

// Quantize 4bit
static PyObject* py_cquantize_blockwise_fp32_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;

    // 파이썬으로부터 전달된 인자 파싱
    // "K": unsigned long long
    // "i": int
    // "KKKKii" : 4개의 K, 2개의 i
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    float* A = reinterpret_cast<float*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    // 템플릿 인스턴스를 호출하는 함수를 호출
    cquantize_blockwise_fp32_fp4(code, A, absmax, out, blocksize, n);

    // 함수가 성공적으로 수행되었음을 파이썬에게 알림
    Py_RETURN_NONE;
}

static PyObject* py_cquantize_blockwise_fp16_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    half* A = reinterpret_cast<half*>(A_ptr); // half
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    cquantize_blockwise_fp16_fp4(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

static PyObject* py_cquantize_blockwise_bf16_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(A_ptr); // bf16
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    cquantize_blockwise_bf16_fp4(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

// Dequantize 4bit
static PyObject* py_cdequantize_blockwise_fp32_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    float* out = reinterpret_cast<float*>(out_ptr);

    cdequantize_blockwise_fp32_fp4(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

static PyObject* py_cdequantize_blockwise_fp16_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    half* out = reinterpret_cast<half*>(out_ptr); // half

    cdequantize_blockwise_fp16_fp4(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

static PyObject* py_cdequantize_blockwise_bf16_fp4(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_ptr); // bf16

    cdequantize_blockwise_bf16_fp4(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

// Quantize blockwise
static PyObject* py_cquantize_blockwise_fp32(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    float* A = reinterpret_cast<float*>(A_ptr); 
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    cquantize_blockwise_fp32(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

static PyObject* py_cquantize_blockwise_fp16(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    half* A = reinterpret_cast<half*>(A_ptr); 
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    cquantize_blockwise_fp16(code, A, absmax, out, blocksize, n);
    
    Py_RETURN_NONE;
}

static PyObject* py_cquantize_blockwise_bf16(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(A_ptr); 
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

    cquantize_blockwise_bf16(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

// Dequantize blockwise
static PyObject* py_cdequantize_blockwise_fp32(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    float* out = reinterpret_cast<float*>(out_ptr); 

    cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}

static PyObject* py_cdequantize_blockwise_fp16(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    half* out = reinterpret_cast<half*>(out_ptr); 

    cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, n);
    
    Py_RETURN_NONE;
}

static PyObject* py_cdequantize_blockwise_bf16(PyObject* self, PyObject* args) {
    unsigned long long code_ptr, A_ptr, absmax_ptr, out_ptr;
    int blocksize, n;
    
    if (!PyArg_ParseTuple(args, "KKKKii", &code_ptr, &A_ptr, &absmax_ptr, &out_ptr, &blocksize, &n)) {
        return NULL;
    }

    float* code = code_ptr == 0 ? NULL : reinterpret_cast<float*>(code_ptr);
    unsigned char* A = reinterpret_cast<unsigned char*>(A_ptr);
    float* absmax = reinterpret_cast<float*>(absmax_ptr);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_ptr); 

    cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, n);

    Py_RETURN_NONE;
}


// ##################################################################################################################################
// Python methods list
static PyMethodDef KbkimLibMethods[] = {
    // METH_VARARGS : 함수가 가변인자를 받음
    {"cgemm_4bit_inference_naive_fp32", py_cgemm_4bit_inference_naive_fp32, METH_VARARGS, "GEMM 4bit inference with fp32"},
    {"cgemm_4bit_inference_naive_fp16", py_cgemm_4bit_inference_naive_fp16, METH_VARARGS, "GEMM 4bit inference with fp16"},
    {"cgemm_4bit_inference_naive_bf16", py_cgemm_4bit_inference_naive_bf16, METH_VARARGS, "GEMM 4bit inference with bf16"},
    
    {"cquantize_blockwise_fp32_fp4", py_cquantize_blockwise_fp32_fp4, METH_VARARGS, "Blockwise quantize with fp32 to fp4"},
    {"cquantize_blockwise_fp16_fp4", py_cquantize_blockwise_fp16_fp4, METH_VARARGS, "Blockwise quantize with fp16 to fp4"},
    {"cquantize_blockwise_bf16_fp4", py_cquantize_blockwise_bf16_fp4, METH_VARARGS, "Blockwise quantize with bf16 to fp4"},

    {"cdequantize_blockwise_fp32_fp4", py_cdequantize_blockwise_fp32_fp4, METH_VARARGS, "Blockwise dequantize with fp4 to fp32"},
    {"cdequantize_blockwise_fp16_fp4", py_cdequantize_blockwise_fp16_fp4, METH_VARARGS, "Blockwise dequantize with fp4 to fp16"},
    {"cdequantize_blockwise_bf16_fp4", py_cdequantize_blockwise_bf16_fp4, METH_VARARGS, "Blockwise dequantize with fp4 to bf16"},

    {"cquantize_blockwise_fp32", py_cquantize_blockwise_fp32, METH_VARARGS, "Blockwise dequantize with fp32"},
    {"cquantize_blockwise_fp16", py_cquantize_blockwise_fp16, METH_VARARGS, "Blockwise dequantize with fp16"},
    {"cquantize_blockwise_bf16", py_cquantize_blockwise_bf16, METH_VARARGS, "Blockwise dequantize with bf16"},

    {"cdequantize_blockwise_fp32", py_cdequantize_blockwise_fp32, METH_VARARGS, "Blockwise dequantize with fp32"},
    {"cdequantize_blockwise_fp16", py_cdequantize_blockwise_fp16, METH_VARARGS, "Blockwise dequantize with fp16"},
    {"cdequantize_blockwise_bf16", py_cdequantize_blockwise_bf16, METH_VARARGS, "Blockwise dequantize with bf16"},

    // 배열의 끝을 알림
    {NULL, NULL, 0, NULL}
};

// Python module definition
static struct PyModuleDef kbkim_lib_module = {
    PyModuleDef_HEAD_INIT, // 모듈 초기화 매크로
    "kbkim_lib",
    NULL, // docstring
    -1, // 모듈 상태 유지를 위한 메모리 미사용
    KbkimLibMethods
};

// Python module initialization
PyMODINIT_FUNC PyInit_kbkim_lib(void) {
    return PyModule_Create(&kbkim_lib_module);
}
