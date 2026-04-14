#!/bin/bash

# Generic options:

#SBATCH --account=bddur45
#SBATCH --time=23:30:00

# Node resources:

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --job-name=compareWANs
#SBATCH -o results/job_gpu-%A_%a.out
#SBATCH -e results/job_gpu-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

# Job array: one task per author file in wan_pairs_authors/ (6 files -> 0-5)
#SBATCH --array=0-5

# Change to project directory so relative paths work
#cd /home/dmitry/Projects/DISKAH/Gabriel/WAN

# Build an array of all wan_pairs_authors txt files
WAN_PAIRS_FILES=(wan_pairs_authors/*.txt)

# Pick the file for this array task
INPUT_FILE=${WAN_PAIRS_FILES[$SLURM_ARRAY_TASK_ID]}

if [ -z "$INPUT_FILE" ]; then
    echo "No input file for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID processing: $INPUT_FILE"

# Example: set this to whatever .IND you want to use for this run
INDICATOR_FILE=${INDICATOR_FILE:-6-authors-whole-plays-top-100-words.IND}

# Activate Conda environment
export CONDADIR=/nobackup/projects/bddur01/$USER
source $CONDADIR/miniconda/etc/profile.d/conda.sh
conda activate cupy-env

time python ./compareWANSnoprint.py -i "$INDICATOR_FILE" "$INPUT_FILE" > "results/$(basename $INPUT_FILE .txt).csv"

echo "End of task $SLURM_ARRAY_TASK_ID"
