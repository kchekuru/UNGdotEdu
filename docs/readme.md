sbatch train.slurm
squeue -u $USER
squeue -j <jobid>
scontrol show job <jobid> | grep -E "JobState|Reason|StartTime|Partition"
tail -f slurm-<jobid>.out [ make sure job is running - ST 'R']

--

# one-time env install/reinstall
pip install --upgrade --force-reinstall -r requirements-cu124.txt

--

# fast GPU validation before full training
sbatch gpu_sanity.slurm
tail -f slurm-gpu-sanity-<jobid>.out

--

ls -lh best_charge_gcn_<RUN_TAG>.pt
ls -lh last_charge_gcn_<RUN_TAG>.pt
tail -50 slurm-<jobid>.out

--
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
tail -50 slurm-<jobid>.out
ls -lh *.pt
