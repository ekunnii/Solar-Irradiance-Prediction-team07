#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate


pip install --no-index tensorflow_gpu==2
pip install --no-index pandas
pip install /project/cq-training-1/project1/teams/team07/lz4-3.0.2-cp37-cp37m-linux_x86_64.whl
pip install --no-index opencv-python
pip install --no-index matplotlib
pip install --no-index tqdm

echo ""
echo "Calling python eval script."
python -u ./evaluator.py "./evaluation_output.csv" "./admin_config.json" -u "./user_config.json" -s "./evaluation_benchmarking.csv"

