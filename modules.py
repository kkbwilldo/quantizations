import math
import types
import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Union
from core import Params4bit, QuantState, dequantize_4bit, gemv_4bit, quantize_kv_2bit, triton_bmm_fA_qB_outer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


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


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None and len(past_key_value) > 0 :
        kv_seq_len += past_key_value[-1]
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    """
    key_states_quant: 채널 축으로 양자화된 캐시. int32
    key_states_full = residual 파트로, 양자화되지 않은 캐시. fp32
    """
    # [bsz, nh, t, hd]
    if past_key_value is not None and len(past_key_value) > 0 :
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        value_states_quant = past_key_value[4]
        value_states_full = past_key_value[5]
        value_scale = past_key_value[6]
        value_mn = past_key_value[7]

        # 양자화된 키 캐시와 쿼리 텐서 간 행렬 곱
        if key_states_quant_trans is not None:
            # fp16으로 dequantize
#             k_cache = unpack_and_dequant_kcache(
#                 key_states_quant_trans, key_scale_trans.unsqueeze(-1), key_mn_trans.unsqueeze(-1), self.group_size, self.k_bits
#             )
#             k_cache = k_cache.transpose(2,3)
#             breakpoint()
#             att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
#                             key_scale_trans, key_mn_trans, self.k_bits)
            att_qkquant = triton_bmm_fA_qB_outer(self.group_size, query_states.to(torch.float16), key_states_quant_trans, 
                            key_scale_trans, key_mn_trans, self.k_bits)
        else:
            att_qkquant = None

        if key_states_full is not None:
            key_states_full = torch.cat([key_states_full, key_states], dim=2)
        else:
            key_states_full = key_states
        att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
        if att_qkquant is not None:
            attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
        else:
            attn_weights = att_qkfull / math.sqrt(self.head_dim)

        if key_states_full.shape[-2] == self.residual_length:
            assert self.residual_length % self.group_size == 0
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = quantize_kv_2bit(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                        self.group_size, 
                                                                                                                        self.k_bits)
            key_states_full = None
            if key_states_quant_trans is not None:
                key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
            else:
                key_states_quant_trans = key_states_quant_trans_new
                key_scale_trans = key_scale_trans_new
                key_mn_trans = key_mn_trans_new

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        value_states_full = torch.cat([value_states_full, value_states], dim=2)
        value_full_length = value_states_full.shape[-2]
        if value_states_quant is None:
            attn_output = torch.matmul(attn_weights, value_states_full)
        else:
#             attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
#                                             value_scale, value_mn, self.v_bits)
            attn_output = triton_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length].to(torch.float16), value_states_quant, 
                                            value_scale, value_mn, self.v_bits)
            attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)
        
        if value_full_length > self.residual_length:
            assert value_full_length == self.residual_length + 1
            value_states_quant_new, scale, mn = quantize_kv_2bit(value_states_full[:, :, :1, :].contiguous(), 
                                                                                            self.group_size, 
                                                                                            self.v_bits)
            value_states_full = value_states_full[:, :, 1:, :].contiguous()
            if value_states_quant is not None:
                value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                value_scale = torch.cat([value_scale, scale], dim=2)
                value_mn = torch.cat([value_mn, mn], dim=2)
            else:
                value_states_quant = value_states_quant_new
                value_scale = scale
                value_mn = mn

    else:
        attn_weights = torch.matmul(query_states, 
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # quantize
        if key_states.shape[-2] % self.residual_length != 0:
            if key_states.shape[-2] < self.residual_length:
                key_states_quant = None
                key_states_full = key_states
            else:
                key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
        else:
            key_states_quant = key_states
            key_states_full = None
        if key_states_quant is not None:
            key_states_quant_trans, key_scale_trans, key_mn_trans = quantize_kv_2bit(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
        else:
            key_states_quant_trans = None
            key_scale_trans = None
            key_mn_trans = None
        
        if value_states.shape[-2] <= self.residual_length:
            value_states_quant = None
            value_states_full = value_states
            value_scale = None
            value_mn = None
        else:
            value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
            value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
            value_states_quant, value_scale, value_mn = quantize_kv_2bit(value_states_quant, 
                                                                                            self.group_size, 
                                                                                            self.v_bits)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
    past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        if self.o_proj.weight.dtype != attn_output.dtype:
            attn_output = attn_output.to(torch.float32)
        attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    
def llama_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][-1]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = hidden_states.to(torch.float16)
    hidden_states = self.norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

   
def apply_kivi(model, num_bits=2, group_size=128, residual_length=128):
    if hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)
    if hasattr(model.model, "forward"):
        model.model.forward = types.MethodType(llama_forward, model.model)
    for layer in model.model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "forward"):
            # Replace the existing forward method with the custom attention_forward_fn
            layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
            layer.self_attn.k_bits = num_bits
            layer.self_attn.v_bits = num_bits
            layer.self_attn.group_size = group_size
            layer.self_attn.residual_length = residual_length
    
    return model
