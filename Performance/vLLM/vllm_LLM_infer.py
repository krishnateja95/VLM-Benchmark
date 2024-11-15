
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
import time
import random
import string
from PIL import Image
import numpy as np
import os, csv

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def generate_random_prompt(length):
    return ''.join(random.choices(string.ascii_letters + ' ', k=length)).strip()

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Demo on using vLLM for offline inference with vision language models')
    
    parser.add_argument('--model-type', '-m', type=str, default="llava", help='Huggingface "model_t0ype".')
    parser.add_argument('--num-prompts', type=int, default=4, help='Number of prompts to run.')
    parser.add_argument('--max-input-len', type=int, default=16, help='Number of Ouput Tokens to generate.')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Number of Ouput Tokens to generate.')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs.')
    parser.add_argument('--warmup-iterations', type=int, default=3, help='Warmup Iterations')
    parser.add_argument('--eval-iterations', type=int, default=3, help='Evaluating epochs')
    args = parser.parse_args()

    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    
    prompts = generate_random_prompt(args.max_input_len)

    sampling_params = SamplingParams(
        # n=args.n,
        temperature=0.0,
        top_p=1.0,
        # use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.max_new_tokens,
    )

    llm = LLM(
        model=args.model_type,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        dtype="float16",
        device="cuda",
        download_dir=cache_dir,
        block_size=16,
        gpu_memory_utilization=0.9,
    )

    # # llm = LLM(model=args.model_type, )
    
    # outputs = llm.generate(prompts, sampling_params)
    
    # warm up stage
    for i in range(args.warmup_iterations):
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    latencies = []
    for i in range(args.eval_iterations):
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    print(f"Avg Latency = {avg_latency}")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")