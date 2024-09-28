import time
import torch
import argparse

from quantizations.modules import apply_kivi
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_title(use_weight_q=False, use_cache_q=False, num_bits_for_kivi=2):
    
    if use_weight_q:
        title = "Quantized model [FP4]"
    else:
        title = "Original model [BF16]"

    if use_cache_q:
        title += f" with quantized kv cache [INT{num_bits_for_kivi}]"
        
    return title


def get_prompt():
    
    instruction_text = """You are a pirate chatbot who always responds in pirate speak!
    User: what did you have for your lunch?
    """
    prompt = f"### Instruction:\n{instruction_text}\n\n### Response:\n"
        
    return prompt


def get_model(model_path, use_weight_q=False, use_cache_q=False, bit_size=2, group_size=128, residual_length=128):

    if use_weight_q:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    if use_cache_q:
        model = apply_kivi(model, bit_size, group_size, residual_length)

    return model


def get_tokenizer(model_path):

    return AutoTokenizer.from_pretrained(model_path)


def parsing():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints_llama3_8b_instruct")
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--num-new-tokens", type=int, default=60)
    parser.add_argument("--num-bits-for-kivi", type=int, default=2)
    parser.add_argument("--group-size-for-kivi", type=int, default=128)
    parser.add_argument("--residual-length-for-kivi", type=int, default=128)
    parser.add_argument("--use-weight-q", action="store_true")
    parser.add_argument("--use-cache-q", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    
    model = get_model(
        args.checkpoint_path, 
        args.use_weight_q, 
        args.use_cache_q, 
        args.num_bits_for_kivi, 
        args.group_size_for_kivi, 
        args.residual_length_for_kivi,
    )

    tokenizer = get_tokenizer(args.checkpoint_path)

    prompt = get_prompt()

    title = get_title(args.use_weight_q, args.use_cache_q, args.num_bits_for_kivi)

    # input text 설정
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.cuda()

    # 텍스트 생성
    max_peak_memory = 0
    avg_time = 0
    for _ in range(args.num_iterations):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_time = time.perf_counter()
        outputs = model.generate(
            input_ids,
            max_new_tokens=args.num_new_tokens, 
            min_new_tokens=args.num_new_tokens, 
            use_cache=True
        )
        end_time = (time.perf_counter() - start_time)
        if _ > 0: avg_time += end_time
        current_peak_memory = torch.cuda.max_memory_allocated()
        if current_peak_memory > max_peak_memory:
            max_peak_memory = current_peak_memory   
     
    if args.num_iterations > 1:
        avg_time /= (args.num_iterations-1)
    else:
        avg_time = end_time
    decode = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"\n\n{title}")
    print(decode[len(prompt):])
    max_peak_memory_mb = max_peak_memory / (1024**3)
    print(f"peak VRAM memory: {max_peak_memory_mb:.2f} GB")
    print(f"TPS : {args.num_new_tokens/avg_time:.4f}\n\n")

if __name__ == "__main__":

    args = parsing()
    main(args)

