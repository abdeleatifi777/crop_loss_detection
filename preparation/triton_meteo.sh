#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH --mem-per-cpu=12G
#SBATCH --array=1-1460


module purge
module load anaconda3
which python
source activate gisthings

csvfile=/scratch/cs/ai_croppro/data/meteo/meteolist2013.txt

x=$SLURM_ARRAY_TASK_ID             
line=`sed "${x}q;d" /scratch/cs/ai_croppro/data/meteo/meteolist2014.txt`
echo $line

srun python extractmeteo.py $line /scratch/cs/ai_croppro/data/shp/rap14_3tiles_barley_reprojected_7800.shp
