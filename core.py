import json
import torch
import operator

#############################################################
# # Nsys 프로파일링 시 필요
# import sys
# sys.path.append("/home/kbkim/4bit_quantization/build")
#############################################################
import kbkim_lib
import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor, device, dtype


name2qmap = {}

dtype2bytes = {}
dtype2bytes[torch.uint8] = 1


class QuantState:
    """양자화 상태 구성 요소를 저장하기 위한 컨테이너 클래스, Params4bit과 함께 사용."""
    
    # 유효한 양자화 타입
    valid_quant_types = ("fp4")
    # 유효한 양자화 상태 키
    valid_qs_keys = [
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state",
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        """
        QuantState 객체를 초기화합니다.

        Parameters:
        - absmax: 각 블록의 절대 최대값.
        - shape: 입력 행렬의 형태.
        - code: 양자화 맵.
        - blocksize: 블록 크기.
        - quant_type: 양자화 타입.
        - dtype: 데이터 타입.
        - offset: 오프셋 값.
        - state2: 중첩된 상태.
        """
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def to(self, device):
        """
        양자화 상태를 device로 옮깁니다.
        
        Parameters:
        - device: 이동할 장치 (CPU 또는 GPU).
        """
        self.absmax = self.absmax.to(device)
        self.offset = self.offset.to(device)
        self.state2.absmax = self.state2.absmax.to(device)
        self.state2.code = self.state2.code.to(device)


class Params4bit(torch.nn.Parameter):
    """
    4비트 양자화된 파라미터를 위한 클래스.

    이 클래스는 4비트 양자화된 파라미터를 저장하고 관리하며, torch.nn.Parameter를 확장합니다.
    """
    def __new__(
        cls,
        data: Optional[Tensor] = None,
        requires_grad=False, # 양자화된 가중치는 기본적으로 고정되어야 함
        quant_state: Optional[QuantState] = None,
        blocksize: int = 64,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module: Optional["Linear4bit"] = None,
        bnb_quantized: bool = False,
    ) -> "Params4bit":
        """
        Params4bit 객체를 생성합니다.

        Parameters:
        - data (Optional[Tensor]): 파라미터 데이터를 포함하는 텐서.
        - requires_grad (bool): 파라미터가 학습 가능한지 여부.
        - quant_state (Optional[QuantState]): 양자화 상태.
        - blocksize (int): 블록 크기.
        - quant_type (str): 양자화 타입.
        - quant_storage (torch.dtype): 양자화 데이터를 저장할 데이터 타입.
        - module (Optional[Linear4bit]): 관련 모듈.
        - bnb_quantized (bool): bnb 양자화 여부.

        Returns:
        - Params4bit: 생성된 Params4bit 객체.
        """

        if data is None:
            data = torch.empty(0)

        self = Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        return self


    def _quantize(self, device):
        """
        매개변수를 4비트 양자화합니다.
        
        Parameters:
        - device: 장치 정보 (CPU 또는 GPU).
        
        Returns:
        - Params4bit: 양자화된 Params4bit 객체.
        """
        w = self.data.contiguous().cuda(device)
        w_4bit, quant_state = quantize_4bit(
            w,
            blocksize=self.blocksize,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
        )
        self.data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self


    def to(self, *args, **kwargs):
        """
        객체를 지정된 장치와 데이터 타입으로 이동합니다.
        
        Parameters:
        - *args, **kwargs: 장치 및 데이터 타입 정보.
        
        Returns:
        - Params4bit: 지정된 장치와 데이터 타입으로 이동된 Params4bit 객체.
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            return self._quantize(device)
        else:
            if self.quant_state is not None:
                self.quant_state.to(device)

            new_param = Params4bit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                quant_type=self.quant_type,
            )

            return new_param


def get_4bit_type(typename, device=None, blocksize=64):
    """
    주어진 타입 이름에 해당하는 4비트 데이터 타입을 생성합니다.

    Parameters:
    - typename (str): 생성할 4비트 데이터 타입의 이름 ("fp4"만 지원됨).
    - device (str 또는 torch.device, optional): 데이터가 저장될 장치 (기본값은 "cuda").
    - blocksize (int): 블록 크기 (사용되지 않음, 호환성을 위해 존재).

    Returns:
    - Tensor: 4비트 데이터 타입을 나타내는 텐서.
    """
    if device is None:
        device = "cuda"
    data = None
    if typename == "fp4":
        # 0b000 = 0
        # 0b001 = 0.0625
        # 0b010 = 8
        # 0b011 = 12
        # 0b100 = 4
        # 0b101 = 6
        # 0b110 = 2
        # 0b111 = 3
        # bnb.functional.create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)로도 생성 가능
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    # 주어진 데이터를 torch 텐서로 변환하고 지정된 장치로 이동
    data = torch.tensor(data, device=device)
    # 데이터 값을 절대값의 최대값으로 나누어 정규화
    data.div_(data.abs().max())

    assert data.numel() == 16

    return data


def get_ptr(A: Optional[Tensor]) -> int:
    """
    텐서의 데이터 포인터를 반환합니다.

    이 함수는 주어진 텐서가 None인 경우 0을 반환하고, 그렇지 않으면 텐서의 데이터 포인터를 반환합니다.

    Parameters:
    - A (Optional[Tensor]): 데이터 포인터를 얻을 텐서.

    Returns:
    - int: 텐서의 데이터 포인터, 텐서가 None인 경우 0.
    """
    if A is None:
        return 0 
    else:
        # 텐서의 데이터 포인터를 반환
        return A.data.data_ptr()


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    동적 양자화 맵을 생성합니다.

    동적 데이터 타입은 동적 지수와 분수로 구성됩니다.
    지수가 0에서 -7로 증가함에 따라 분수에 사용 가능한 비트 수가 줄어듭니다.

    이는 동적 타입의 일반화로, 일부 비트를 선형 양자화 영역(분수)으로 예약할 수 있습니다.
    n은 지수 비트의 최대 수를 결정합니다.

    자세한 내용은 다음을 참조하십시오:
    [8-Bit Approximations for Parallelism in Deep Learning](https://arxiv.org/abs/1511.04561)

    Parameters:
    - signed (bool): 부호가 있는 데이터 타입을 생성할지 여부.
    - max_exponent_bits (int): 지수 비트의 최대 수.
    - total_bits (int): 총 비트 수.

    Returns:
    - Tensor: 동적 양자화 맵을 포함하는 텐서.
    """
    data = []
    # 추가 항목: 모든 지수 비트가 0이고 indicator가 없는 경우
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        # 분수 항목의 수 계산
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        # 0.1에서 1 사이를 fraction_items 개수만큼 균등하게 나눔
        boundaries = torch.linspace(0.1, 1, fraction_items)
        # 경계 값의 평균 계산
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        # 동적 지수와 분수를 곱한 값을 데이터에 추가
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        # 추가 항목이 있는 경우
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    # 데이터의 길이가 총 비트 수의 제곱과 같은지 확인
    assert len(data) == 2**total_bits

    # 데이터 길이와 256 사이의 차이 계산
    gap = 256 - len(data)
    for i in range(gap):
        data.append(0) # 차이를 0으로 채움

    # 데이터를 정렬
    data.sort()
    # 텐서로 변환하여 반환
    return Tensor(data)


def quantize_blockwise(
    A: Tensor,
    blocksize=4096,
) -> Tuple[Tensor, QuantState]:
    """
    텐서 A를 블록 단위로 양자화합니다.

    텐서 A를 4096 값의 블록으로 나누어 양자화합니다.
    그런 다음 이러한 블록 내에서 절대 최대값을 계산하여 비선형 양자화를 수행합니다.

    Parameters:
    - A (Tensor): 입력 텐서.
    - blocksize (Optional[int]): 블록 크기 (기본값 4096).

    Returns:
    - Tensor: 8비트 텐서.
    - QuantState: 양자화 상태, 양자화를 원래 상태로 되돌리기 위한 상태.
    """
    # 양자화 맵 생성
    if "dynamic" not in name2qmap:
        name2qmap["dynamic"] = create_dynamic_map().to(A.device)
    code = name2qmap["dynamic"]

    # 절대 최대값 초기화
    n = A.numel()
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

    # 출력 텐서 초기화
    out = torch.zeros_like(A, dtype=torch.uint8)

    # 유효한 블록 크기 확인
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
    blocksize = blocksize
    code = code.to(A.device)

    # custom kernel 호출
    kbkim_lib.cquantize_blockwise_fp32(
        get_ptr(code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        blocksize,
        A.numel(),
    )

    quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=A.dtype)

    return out, quant_state


def dequantize_blockwise(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    code: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 4096,
    nested=False,
) -> Tensor:
    """
    블록 단위로 양자화된 값을 복원합니다.

    4096 크기의 블록에서 최대 절대값 absmax를 사용하여 텐서 A를 복원합니다.

    Parameters: 
    - A (Tensor): 입력 8비트 텐서.
    - quant_state (QuantState): 코드, absmax 및 기타 양자화 상태 구성 요소를 포함하는 객체.
    - absmax (Tensor): 최대 절대값.
    - code ( Tensor): 양자화 맵.
    - out (Tensor): 복원된 출력 텐서 (기본값: float32).
    - blocksize (int): 블록 크기 (기본값: 4096).
    - nested (bool): 중첩 양자화 여부 (기본값: False).

    Returns:
    - Tensor: 복원된 텐서 (기본값: float32).
    """
    # quant_state나 absmax 중 하나는 반드시 제공되어야 함
    assert quant_state is not None or absmax is not None

    absmax = quant_state.absmax

    # 출력 텐서 초기화
    if out is None:
        out = torch.empty(A.shape, dtype=quant_state.dtype, device=A.device)

    # 양자화 맵 장치로 이동
    code = quant_state.code.to(A.device)

    # 유효한 블록 크기 확인
    if quant_state.blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
        raise ValueError(
            f"The blockwise of {quant_state.blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]",
        )

    # custom kernel 호출
    kbkim_lib.cdequantize_blockwise_fp32(
        get_ptr(quant_state.code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        quant_state.blocksize,
        A.numel(),
    )

    return out


def gemv_4bit(
    A: Tensor,
    B: Tensor,
    out: Optional[Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state=None,
):
    """
    4비트 양자화된 행렬 곱셈을 수행합니다.

    4비트 양자화된 행렬 B와 벡터 A의 곱셈을 수행하여 결과를 out 텐서에 저장합니다.
    이 함수는 양자화된 상태를 필요로 하며, 양자화된 상태가 제공되지 않으면 예외를 발생시킵니다.

    Parameters:
    - A (Tensor): 입력 벡터.
    - B (Tensor): 4비트 양자화된 행렬.
    - out (Tensor, optional): 결과를 저장할 출력 텐서.
    - transposed_A (bool, optional): A의 전치 여부 (기본값: False).
    - transposed_B (bool, optional): B의 전치 여부 (기본값: False).
    - state: 양자화 상태 정보.

    Returns:
    - Tensor: 4비트 양자화된 행렬-벡터 곱셈 결과 텐서.
    """

    # 양자화 상태가 None이면 예외 발생
    if state is None:
        raise ValueError("state cannot None. gem_4bit( ) requires the state from quantize_4bit( )")

    # 입력 벡터 A의 차원이 올바르지 않으면 예외 발생
    if A.numel() != A.shape[-1]:
        raise ValueError(
            'Dimensions of A are invalid. Must be a vector with the leading dimensions of "1", e.g. [1, 1, 2048]',
        )

    Bshape = state.shape
    bout = Bshape[0]
    absmax = state.absmax

    # 중첩된 양자화 상태 처리
    absmax = dequantize_blockwise(state.absmax, state.state2)
    absmax += state.offset

    # 출력 텐서가 None이면 초기화
    if out is None:
        if len(A.shape) == 3:
            out = torch.empty(size=(A.shape[0], A.shape[1], bout), dtype=A.dtype, device=A.device)
        else:
            out = torch.empty(size=(A.shape[0], bout), dtype=A.dtype, device=A.device)

    n = 1
    m = Bshape[0]
    k = Bshape[1]
    lda = Bshape[0]
    ldc = Bshape[0]
    ldb = (A.shape[-1] + 1) // 2

    # custom kernel 호출
    if B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32]:
        kbkim_lib.cgemm_4bit_inference_naive_fp32(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(state.code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            state.blocksize,
        )

    else:
        raise NotImplementedError(f"Matmul not implemented for data type {A.dtype}")

    return out


def quantize_4bit(
    A: Tensor,
    blocksize=64,
    quant_type="fp4",
    quant_storage=torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """
    텐서 A를 4비트 값으로 블록 단위 양자화합니다.

    텐서 A를 블록으로 나누어 각각 독립적으로 FP4로 양자화합니다.

    Parameters:
    - A (Tensor): 입력 텐서.
    - blocksize (int): 양자화에 사용될 블록 크기.
    - quant_type (Optional[str]): 4비트 양자화 데이터 타입 {fp4} (기본값: "fp4").
    - quant_storage (Optional[torch.dtype]): 양자화된 값을 저장할 데이터 타입 (기본값: torch.uint8).

    Returns:
    - Tensor: 4비트 값으로 패킹된 텐서.
    - QuantState: 양자화를 원래 상태로 되돌리기 위한 양자화 상태.
    """

    # 입력 텐서가 CUDA 장치에 있는지 확인
    if A.device.type != "cuda":
        raise NotImplementedError(f"Device type not supported for FP4 quantization: {A.device.type}")
    # 지원되지 않는 양자화 타입 확인
    if quant_type not in ["fp4"]:
        raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")

    n = A.numel()
    input_shape = A.shape

    # 최대 절대값 텐서 초기화
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

    # 출력 텐서 초기화
    mod = dtype2bytes[quant_storage] * 2 # 1*2
    out = torch.zeros(((n + 1) // mod, 1), dtype=quant_storage, device=A.device)

    # 유효한 블록 크기 확인
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]

    # custom kernel 호출
    kbkim_lib.cquantize_blockwise_fp16_fp4(
        get_ptr(None),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        blocksize,
        n,
    )

    code = get_4bit_type(quant_type, device=A.device)

    offset = absmax.mean()
    absmax -= offset
    qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
    del absmax
    state = QuantState(
        absmax=qabsmax,
        shape=input_shape,
        dtype=A.dtype,
        blocksize=blocksize,
        code=code,
        quant_type=quant_type,
        offset=offset,
        state2=state2,
    )

    return out, state


def dequantize_4bit(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    blocksize: int = 64,
    quant_type="fp4",
) -> Tensor:
    """
    4비트 블록 단위로 양자화된 값을 복원합니다.

    블록 크기 blocksize로 최대 절대값 absmax를 사용하여 텐서 A를 복원합니다.

    Parameters:
    - A (Tensor): 입력 텐서 (패킹된 4비트 값).
    - quant_state (Optional[QuantState]): 양자화 통계(절대 최대값, 원래 텐서 형태 및 원래 데이터 타입 포함)를 가진 객체.
    - blocksize (int): 양자화에 사용된 블록 크기.
    - quant_type (str): 4비트 양자화 데이터 타입 {fp4}.

    Returns:
    - Tensor: 복원된 텐서.
    """

    # 유효한 블록 크기 확인
    if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
        raise ValueError(
            f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]",
        )
    # 지원되지 않는 양자화 타입 확인
    if quant_type not in ["fp4"]:
        raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")

    absmax = quant_state.absmax

    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

    n = out.numel()

    # custom kernel 호출
    kbkim_lib.cdequantize_blockwise_fp16_fp4(
        get_ptr(None),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        quant_state.blocksize,
        n,
    )

    # 텐서를 transpose하여 반환
    return out.t() 
