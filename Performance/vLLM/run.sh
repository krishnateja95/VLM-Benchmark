#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=30:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate trt_llm_vllm

export TMPDIR="/lus/grand/projects/datascience/krishnat/tmp_dir/"
export RAY_TMPDIR='/tmp'

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/VLM_Inference_Bench/VLM-Benchmark/Performance/vLLM/

for model in "llava"; do
    for tensor_parallel in 1; do
        for batch_size in 1; do
            for image_size in 128; do
                for input_output_length in 128; do
                    python3 vllm_VLM_infer.py --model-type=$model --num-prompts=$batch_size --max-input-len=$input_output_length --max-new-tokens=$input_output_length --image-size=$image_size --num-gpus=$tensor_parallel --modality="image"
                done
            done
        done
    done
done


# for model in "meta-llama/Llama-2-7b-hf"; do
#     for tensor_parallel in 1; do
#         for batch_size in 1; do
#             for input_output_length in 128; do
#                 python3 vllm_LLM_infer.py --model-type=$model --num-prompts=$batch_size --max-input-len=$input_output_length --max-new-tokens=$input_output_length --num-gpus=$tensor_parallel
#             done
#         done
#     done
# done
