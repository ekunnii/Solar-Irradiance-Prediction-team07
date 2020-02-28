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
pip install --no-index pytz
pip install /project/cq-training-1/project1/teams/team07/timezonefinder-4.2.0-py3-none-any.whl

cp -ru /project/cq-training-1/project1/teams/team07/.keras ~/

# python --version
# which python
# which pip
pip freeze

echo ""
echo "Calling python train script."
#stdbuf -oL python -u ./train.py $PWD/train_config.json $PWD/train_config.json -n 10 -m "CNN2D" --scratch_dir $SCRATCH --delete_checkpoints --use_cache -s 123
#stdbuf -oL python -u ./train.py $PWD/train_config.json $PWD/train_config.json -n 10 -m "pretrained_resnet" --scratch_dir $SCRATCH --delete_checkpoints --use_cache -s 123
#stdbuf -oL python -u ./train.py $PWD/train_config.json $PWD/train_config.json -n 10 -m "cnn_lstm" --scratch_dir $SCRATCH --delete_checkpoints --use_cache -s 1234 --user_config user_config.json --save_best

#stdbuf -oL python -u ./train.py $PWD"/train_config.json" -n 200 -m "double_pretrained_resnet" --run_setting "double pretrained resnet50, crop 64, nofreeze" --scratch_dir $SCRATCH --load_checkpoints
stdbuf -oL python -u ./train.py $PWD"/train_config.json" -n 50 -m "cnn_lstm" --scratch_dir $SCRATCH --run_setting "last minute cnn-lstm" --user_config user_config.json --save_best --use_cache --load_checkpoints
#stdbuf -oL python -u ./train.py $PWD"/train_config.json" -n 30 -m "pretrained_resnet" --run_setting "pretrained resnet50, crop 64, nofreeze" --scratch_dir $SCRATCH --use_cache --load_checkpoints --save_best
# stdbuf -oL python -u ./train.py $PWD"/train_config.json" -n 50 -m "double_cnn_lstm" --run_setting "double cnn double! lstm, full meta, crop 64, lstm(128), T0-1h T0-2h, freeze 50%" --scratch_dir $SCRATCH --use_cache --user_config user_config.json --save_best
