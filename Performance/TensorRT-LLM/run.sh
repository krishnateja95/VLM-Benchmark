module use /soft/modulefiles/
module load conda
conda activate trt_llm_vllm

python3 trt_llm_infer.py --model-type="mllama" --num-prompts=4 --modality="image"

# python3 vllm_infer.py --model-type="mllama" --num-prompts=4 --modality="image"


