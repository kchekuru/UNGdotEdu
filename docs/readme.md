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

--

# model artifacts (hugging face hub)

This repository on GitHub contains training code, configs, and documentation.
Model artifacts are published on Hugging Face Hub.

Hugging Face model repo:
https://huggingface.co/kchek2546/UNGdotEdu1

Direct file links:
- Weights (.pt): https://huggingface.co/kchek2546/UNGdotEdu1/resolve/main/model_final_best.pt
- Config (.json): https://huggingface.co/kchek2546/UNGdotEdu1/resolve/main/model_final_best.config.json

Upload commands:
hf upload kchek2546/UNGdotEdu1 trained-model/model_final_best.pt model_final_best.pt
hf upload kchek2546/UNGdotEdu1 trained-model/model_final_best.config.json model_final_best.config.json
