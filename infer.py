import torch
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM

from huggingface_hub import login
login("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")

def batch_encode(prompts, tokenizer):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=len(prompts))
    for t in input_tokens:
      if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    return input_tokens


def generate_prompt(model, tokenizer, prompts):

  input_tokens = batch_encode(prompts, tokenizer)
  print(input_tokens)
  generate_kwargs = dict(max_new_tokens=30, do_sample=False)
  output_ids = model.generate(**input_tokens, **generate_kwargs)
  print(output_ids)
  outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

  return outputs

if __name__ == '__main__':
    model_name = "meta-llama/Llama-2-7b-hf"
    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    model = LlamaForCausalLM.from_pretrained(model_name,
                                             cache_dir   = cache_dir,
                                             torch_dtype = torch.float16,
                                             device_map  = 'auto'
                                             )
    model.seqlen = 4096
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer)
    exit()
    tokenizer.pad_token = tokenizer.eos_token
    output = generate_prompt(model, tokenizer, prompts=["London is the Capital of "])

    print(output)
