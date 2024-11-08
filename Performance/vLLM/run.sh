#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=30:00:00
#PBS -q preemptable
#PBS -A datascience

# module use /soft/modulefiles/
# module load conda
# conda activate trt_llm_vllm

# export TMPDIR="/lus/grand/projects/datascience/krishnat/tmp_dir/"
# export RAY_TMPDIR='/tmp'

# cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/VLM_Inference_Bench/VLM-Benchmark/Performance/vLLM/

# for model in "mllama" "pixtral_hf" "molmo" "qwen2_vl" "llava" "phi3_v"; do
#     for tensor_parallel in 1 2 4; do
#         for batch_size in 1 16 32 64 128; do
#             for image_size in 128 256 512 1024; do
#                 for input_output_length in 128 256 512 1024 2048; do
#                     python3 vllm_infer.py --model-type=$model --num-prompts=$batch_size --max-input-len=$input_output_length --max-new-tokens=$input_output_length --image-size=$image_size --num-gpus=$tensor_parallel --modality="image"
#                 done
#             done
#         done
#     done
# done



cp mllama_infer.py chameleon_infer.py
cp mllama_infer.py llava_infer.py
cp mllama_infer.py molmo_infer.py
cp mllama_infer.py paligemma_infer.py
cp mllama_infer.py phi3v_infer.py
cp mllama_infer.py pixtral_infer.py
cp mllama_infer.py qwen2vl_infer.py