#!/bin/bash
#SBATCH -t 03-00
#SBATCH --mem=12G
#SBATCH -o /scratch/cs/ai_croppro/data/samantha/out.out
#SBATCH -e /scratch/cs/ai_croppro/data/samantha/error.err
#SBATCH --array=0-80

module purge
module load anaconda3
which python
source activate gisthings
which python


for myfile in /scratch/cs/ai_croppro/data/ls7lists/* ; do
 
  echo $filename
  srun ~/.conda/envs/gisthings/bin/python arrayextractor.py $myfile 
 
done
  
