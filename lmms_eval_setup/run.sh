module use /soft/modulefiles/
module load conda

conda activate lmms_eval

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

python3 -m accelerate.commands.launch --num_processes=16 main.py \
    --model llama_vision \
    --model_args pretrained="meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --tasks refcoco \
    --batch_size 1 \