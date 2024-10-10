module use /soft/modulefiles/
module load conda

conda create -n VLMs python=3.11 -y 
conda activate VLMs

pip install -r requirements.txt