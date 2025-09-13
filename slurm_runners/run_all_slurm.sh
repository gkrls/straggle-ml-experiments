# run all scripts in current folder except this one
for script in $(ls $(dirname "$0") | grep -E 'run_.*\.sh' | grep -v 'run_all_slurm.sh'); do
  sbatch "$(dirname "$0")/$script"
done