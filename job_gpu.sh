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

# Job array: one task per file in wan_pairs/
# The range must match the number of files; adjust upper bound as needed
#SBATCH --array=0-8%10   # Replace N with (number of files - 1)

# Change to project directory so relative paths work
cd /home/dmitry/Projects/DISKAH/Gabriel/WAN

# Build an array of all wan_pairs txt files
WAN_PAIRS_FILES=(wan_pairs_authors/*.txt)

# Pick the file for this array task
INPUT_FILE=${WAN_PAIRS_FILES[$SLURM_ARRAY_TASK_ID]}

if [ -z "$INPUT_FILE" ]; then
    echo "No input file for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID processing: $INPUT_FILE"

mkdir -p results

time python ./compareWANSnoprint.py "$INPUT_FILE" > "results/$(basename $INPUT_FILE .txt).csv"

echo "End of task $SLURM_ARRAY_TASK_ID"
