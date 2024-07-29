cmake_minimum_required(VERSION 3.10)
project(kbkim_lib CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA 설정
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Python 설정
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# 추가 헤더 파일 경로
include_directories(${CMAKE_SOURCE_DIR}/csrc)

# CUDA 소스 파일 추가
set(CUDA_SOURCE_FILES 
    csrc/ops.cu
    csrc/kernels.cu
)

# CUDA 컴파일 옵션 설정
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_86;--compiler-options -fPIC)

# CUDA 라이브러리 생성
cuda_add_library(cuda_ops STATIC ${CUDA_SOURCE_FILES})

# Python 모듈 생성
Python_add_library(kbkim_lib MODULE pythonInterface.cpp)

# CUDA 라이브러리와 Python 모듈 연결
target_link_libraries(kbkim_lib PRIVATE 
    cuda_ops
    ${CUDA_LIBRARIES}
)

# 빌드된 라이브러리 이름 설정
set_target_properties(kbkim_lib PROPERTIES PREFIX "")
set_target_properties(kbkim_lib PROPERTIES SUFFIX ".so")
