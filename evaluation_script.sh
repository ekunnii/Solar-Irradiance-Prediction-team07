#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate


pip install --no-index tensorflow_gpu==2
pip install --no-index pandas
pip install /project/cq-training-1/project1/teams/team07/lz4-3.0.2-cp37-cp37m-linux_x86_64.whl
pip install /project/cq-training-1/project1/teams/team07/timezonefinder-4.2.0-py3-none-any.whl
pip install --no-index opencv-python
pip install --no-index matplotlib
pip install --no-index tqdm
pip install --no-index pytz
pip install /project/cq-training-1/project1/teams/team07/timezonefinder-4.2.0-py3-none-any.whl

echo ""
echo "Calling python eval script."
python -u /project/cq-training-1/project1/submissions/team07/code/evaluator.py "/project/cq-training-1/project1/submissions/team07/code/evaluation_output.csv" "/project/cq-training-1/project1/submissions/team07/code/admin_config.json" -u "/project/cq-training-1/project1/submissions/team07/code/user_config.json" -s "/project/cq-training-1/project1/submissions/team07/code/evaluation_benchmarking.csv"



