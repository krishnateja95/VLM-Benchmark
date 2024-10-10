module use /soft/modulefiles/
module load conda
conda activate VLMs

python3 vllm_infer.py --model-type="mllama" --num-prompts=4 --modality="image"

