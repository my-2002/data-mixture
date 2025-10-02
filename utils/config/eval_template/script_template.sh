#!/bin/bash
#SBATCH -J {model_name}
#SBATCH -o job-{model_name}.log
#SBATCH -e job-{model_name}.err
#SBATCH -p GPU-8A100
#SBATCH -N 1 -n 4
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
. /etc/profile.d/modules.sh
module load cuda/12.4
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu core.

opencompass examples/data_mixture/eval_{model_name}.py -a vllm