module use /soft/modulefiles/
module load conda
conda activate trt_llm_vllm

# for model in "llava" "llava-next" "llava-onevision" "fuyu" "phi3_v" "paligemma" "chameleon" "minicpmv" "blip-2" "internvl_chat" "NVLM_D" "qwen_vl" "qwen2_vl" "mllama"; do
for model in "mllama"; do
    python3 vllm_infer.py --model-type=$model --num-prompts=4 --modality="image"
done


