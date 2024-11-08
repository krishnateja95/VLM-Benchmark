
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

# LLaVA-1.5
def run_llava(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "llava-hf/llava-1.5-7b-hf"
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096, download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name

def get_llava_data(question: str):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    stop_token_ids = None
    return prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "microsoft/Phi-3-vision-128k-instruct"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        download_dir=cache_dir,
        mm_processor_kwargs={"num_crops": 16},
        tensor_parallel_size=num_gpus
    )
    return llm, model_name

def get_phi3v_data(question: str):
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    stop_token_ids = None
    return prompt, stop_token_ids


# PaliGemma
def run_paligemma(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "google/paligemma-3b-mix-224"
    llm = LLM(model=model_name, download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name

def get_paligemma_data(question: str):
    prompt = "caption en"
    stop_token_ids = None
    return prompt, stop_token_ids



# Chameleon
def run_chameleon(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "facebook/chameleon-7b"
    llm = LLM(model=model_name, max_model_len=4096, download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name

def get_chameleon_data(question: str):
    prompt = f"{question}<image>"
    stop_token_ids = None
    return prompt, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "mistral-community/pixtral-12b"
    llm = LLM(model=model_name, max_model_len=8192, download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name

def get_pixtral_hf_data(question: str):
    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    stop_token_ids = None
    return prompt, stop_token_ids


# Molmo
def run_molmo(modality, num_gpus: int):
    assert modality == "image"
    model_name = "allenai/Molmo-7B-D-0924"
    llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=num_gpus)
    return llm, model_name

def get_molmo_data(question):
    prompt = question
    stop_token_ids = None
    return prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    llm = LLM(model=model_name, max_model_len=8192, max_num_seqs=5,download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name

def get_qwen2_vl_data(question: str):
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return prompt, stop_token_ids



# LLama 3.2
def run_mllama(modality: str, num_gpus: int):
    assert modality == "image"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = LLM(model=model_name, max_model_len=4096, max_num_seqs=16, enforce_eager=True, download_dir=cache_dir, tensor_parallel_size=num_gpus)
    return llm, model_name


def get_mllama_data(question: str):
    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return prompt, stop_token_ids



model_example_map = {
    "llava": run_llava,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "qwen2_vl": run_qwen2_vl,
    "mllama": run_mllama,
    "pixtral_hf": run_pixtral_hf,
    "molmo": run_molmo
}

model_example_data_map = {
    "llava": get_llava_data,
    "phi3_v": get_phi3v_data,
    "paligemma": get_paligemma_data,
    "chameleon": get_chameleon_data,
    "qwen2_vl": get_qwen2_vl_data,
    "mllama": get_mllama_data,
    "pixtral_hf": get_pixtral_hf_data,
    "molmo": get_molmo_data
}


file_name_dict = {
    "llava": "llava_1_5.csv",
    "phi3_v": "phi3v.csv",
    "paligemma": "paligemma.csv",
    "chameleon": "chameleon.csv",
    "qwen2_vl": "qwen2_vl.csv",
    "mllama": "mllama.csv",
    "pixtral_hf": "pixtral_hf.csv",
    "molmo": "molmo.csv"
    }


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        
        def generate_blank_image(size):
            return Image.new('RGB', (size, size), (255, 255, 255))

        def generate_random_prompt(length):
            return ''.join(random.choices(string.ascii_letters + ' ', k=length)).strip()

        img_question = generate_random_prompt(args.max_input_len)
        image        = generate_blank_image(args.image_size)

        # image = ImageAsset("cherry_blossom") \
        #     .pil_image.convert("RGB")
        # img_question = "What is the content of this image?"
        
        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)



def run_generation(args, llm, model, model_name, modality):
    mm_input = get_multi_modal_input(args)
    
    data = mm_input["data"]
    question = mm_input["question"]
    
    prompt, stop_token_ids = model_example_data_map[model](question)

    sampling_params = SamplingParams(temperature=0.2,
                                    max_tokens=args.max_new_tokens,
                                    stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        } for _ in range(args.num_prompts)]

    # warm up stage
    for i in range(args.warmup_iterations):
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    latencies = []
    for i in range(args.warmup_iterations):
        start_time = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    
    text_tokens = args.num_prompts*(args.max_input_len + args.max_new_tokens) 
    image_tokens = args.num_prompts*(int(args.image_size*args.image_size))
    
    multimodal_tokens = image_tokens + text_tokens 
    
    multimodal_token_throughput =  multimodal_tokens/avg_latency
    multimodal_request_throughput = args.num_prompts/avg_latency
    
    list_1 = ["Hardware", "Num of Hardware", "Framework", "Model", "Input Length", "Output Length", "Image Size", "Batch Size",
            "Text Tokens", "Image Tokens", "Multimodal Tokens",
            "Avg Latency", "Multimodal Token Throughput", "Multimodal Request Throughput"]
    
    list_2 = ["Nvidia A100 GPU", args.num_gpus, "vLLM", model_name, args.max_input_len, args.max_new_tokens, args.image_size, args.num_prompts,
            text_tokens, image_tokens, multimodal_tokens, 
            avg_latency, multimodal_token_throughput, multimodal_request_throughput] 
    

    assert len(list_1) == len(list_2)

    csv_file = "Results/" + file_name_dict[args.model_type]
    
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(list_1)
        
        writer.writerow(list_2) 
        
    csvfile.close()


def main(args):
    
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    llm, model_name = model_example_map[model](modality, args.num_gpus)

    for batch_size in [1, 16, 32, 64]:
        for image_size in [128, 256, 512, 1024]:
            for input_output_length in [128, 256, 512, 1024]:
                args.num_prompts    = batch_size
                args.image_size     = image_size
                args.max_input_len  = input_output_length
                args.max_new_tokens = input_output_length
                run_generation(args, llm, model, model_name, modality)


    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(len(generated_text))
    #     print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Demo on using vLLM for offline inference with vision language models')
    
    parser.add_argument('--model-type', '-m', type=str, default="llava", choices=model_example_map.keys(), help='Huggingface "model_t0ype".')
    
    parser.add_argument('--num-prompts', type=int, default=4, help='Number of prompts to run.')
    
    parser.add_argument('--max-input-len', type=int, default=16, help='Number of Ouput Tokens to generate.')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Number of Ouput Tokens to generate.')
    
    parser.add_argument('--image-size', type=int, default=256, help='Size of the input Image.')
    
    parser.add_argument('--modality', type=str, default="image", choices=['image', 'video'], help='Modality of the input.')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to extract from the video.')

    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs.')

    parser.add_argument('--warmup-iterations', type=int, default=3, help='Warmup Iterations')
    parser.add_argument('--eval-iterations', type=int, default=3, help='Evaluating epochs')

    args = parser.parse_args()
    main(args)


