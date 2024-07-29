import torch
import torch.nn as nn
from core import Params4bit, QuantState, dequantize_4bit, gemv_4bit

"""
해당 quantization을 사용하는 Linear layer는 inference 전용 모듈입니다.
따라서 autograd.Function을 상속받아 커스텀하는 함수 클래스를 만들어 forward와 backward를 구현하지 않고,
1) torch.nn.functional.linear 함수와 2) custom kernel을 사용해 forward를 수행합니다.

1)을 사용하는 경우는 prefill phase와 같이 seq_len이 1을 초과하는 경우 혹은 배치 사이즈가 1을 초과하는 경우이며,
2)를 사용하는 경우는 decoding phase와 같이 seq_len이 1인 경우입니다.

1)의 경우에는 양자화된 가중치를 dequantize하여 matmul을 수행합니다.

--------------------------------------------------------------------------------------------------------------

Linear4bit 모듈은 transformers의 interface를 그대로 사용할 수 있으며, 사용법은 아래와 같습니다.
- transformers/integrations/bitsandbytes.py(_replace_with_bnb_linear)의 bnb.nn.Linear4bit을 본 프로젝트의 Linear4bit으로 교체하여 사용할 수 있습니다.
- bnb.nn.Linear4bit의 interface에 맞춰 Linear4bit을 구현하였고, 내부 구현은 동일하지 않습니다.
- Llama3-8B-Instruct 모델에 한해서 모듈에 대한 테스트가 진행되었습니다.

--------------------------------------------------------------------------------------------------------------

현재 bnb의 Linear4bit의 cuda kernel이 본 프로젝트 cuda kernel의 약 절반의 latency를 가집니다.
이에 대해 kernel 분석 중입니다...
"""

def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state: QuantState,
    out: torch.Tensor = None,
    bias=None,
):
    """
    4비트 양자화된 행렬 곱셈을 수행합니다.

    양자화된 행렬 B와 입력 행렬 A의 곱셈을 수행하고, 필요에 따라 바이어스를 더합니다.
    벡터-행렬 곱셈 또는 일반 행렬 곱셈을 수행할 수 있습니다.

    Parameters:
    - A (torch.Tensor): 입력 텐서.
    - B (torch.Tensor): 4비트 양자화된 텐서.
    - quant_state (QuantState): 양자화 상태 객체.
    - out (torch.Tensor): 출력 텐서.
    - bias (torch.Tensor): 바이어스 텐서.

    Returns:
    - torch.Tensor: 행렬 곱셈 결과 텐서.
    """
    assert quant_state is not None

    # Decoding phase
    # 입력 텐서의 형식이 (1,1,hidden_size) 형식인 경우. 
    # 이 경우 입력 텐서는 사실상 1D Tensor
    if A.numel() == A.shape[-1]:
        # vector-matrix multiplication을 수행 
        out = gemv_4bit(A, B.t(), out, state=quant_state)
        if bias is not None:
            out += bias
        return out
    # Prefill phase
    else:
        return torch.nn.functional.linear(A, dequantize_4bit(B, quant_state).to(A.dtype).t(), bias) 


class Linear4bit(nn.Linear):

    """
    4비트 양자화를 사용하는 선형 레이어 클래스.

    이 클래스는 4비트 양자화된 가중치를 사용하는 선형 레이어를 구현합니다.
    입력과 가중치의 곱셈 및 덧셈 연산을 수행하며, 필요에 따라 양자화 상태를 복원합니다.

    Parameters:
    - input_features (int): 입력 피처의 수.
    - output_features (int): 출력 피처의 수.
    - bias (bool): 바이어스를 사용할지 여부 (기본값: False). # Llama3는 bias를 사용하지 않음
    - compute_dtype (torch.dtype): 연산에 사용할 데이터 타입 (기본값: None).
    - compress_statistics (bool): 통계 정보를 압축할지 여부 (기본값: False). # bnb 인터페이스에 맞추기 위함일 뿐 실제로 사용하지 않음
    - quant_type (str): 4비트 양자화 데이터 타입 (기본값: "fp4").
    - quant_storage (torch.dtype): 양자화된 값을 저장할 데이터 타입 (기본값: torch.uint8).
    - device (torch.device): 연산에 사용할 장치 (기본값: None).
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=False,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        quant_storage=torch.uint8,
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        
        # 4비트 양자화된 가중치 파라미터를 생성
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False
        self.quant_state = None
        self.quant_storage = quant_storage

    def set_compute_type(self, x):
        """
        입력 데이터 타입에 따라 연산에 사용할 데이터 타입을 설정합니다.

        Parameters:
        - x (torch.Tensor): 입력 텐서.
        """

        if x.dtype in [torch.float32, torch.bfloat16]:
            # 입력 데이터 타입이 연산에 안전한 경우, 해당 타입으로 설정하여 연산 속도 및 안정성을 확보
            self.compute_dtype = x.dtype

    def forward(self, x: torch.Tensor):
        """
        순전파 연산을 수행합니다.

        Parameters:
        - x (torch.Tensor): 입력 텐서.

        Returns:
        - torch.Tensor: 출력 텐서.
        """

        # 연산에 사용할 데이터 타입 설정
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        # 4비트 양자화된 가중치를 사용하여 입력과의 곱셈 및 바이어스 덧셈 연산 수행
        out = matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)

        # 출력 텐서의 데이터 타입을 입력 텐서의 데이터 타입으로 변환
        out = out.to(inp_dtype)

        return out
