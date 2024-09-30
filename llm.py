import torch
from huggingface_hub import login

from transformers import AutoTokenizer

login("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")

def batch_encode(prompts, tokenizer, prompt_len=512):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=prompt_len)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens


def generate_prompt(model, tokenizer, prompts):
    
    input_tokens = batch_encode(prompts, tokenizer)

    generate_kwargs = dict(max_new_tokens=32, do_sample=False)
    output_ids = model.generate(**input_tokens, **generate_kwargs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs



if __name__ == '__main__':

    from opt_model import OPTForCausalLM
    model_name = "facebook/opt-125m"
    
    cache_dir = None
    model = OPTForCausalLM.from_pretrained(model_name,
                                             cache_dir   = cache_dir,
                                             torch_dtype = torch.float16,
                                            #  device_map  = 'auto'
                                             )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    output = generate_prompt(model, tokenizer, prompts=["London is the Capital of "])

    print(output)