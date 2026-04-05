4/3/2026
Final Test MSE: 1.848888 based on training job 3811753, which is high and barely moved from initial MSE, 5 layers and 200 epochs(hard coding) 
Ran on PSC - CPU
--
Removed hard coding for layers and epochs
Downgraded PyTorch from 2.11 to 2.6 and forcing run's on GPU
Ran on PSC - GPU

sbatch --export=ALL,RUN_TAG=runA,RESUME=0,NUM_LAYERS=2,HIDDEN_CHANNELS=256,DROPOUT=0.05,LR=2e-4,NUM_EPOCHS=80,LR_PATIENCE=4,EARLY_STOP_PATIENCE=12 train-quick.slurm
sbatch --export=ALL,RUN_TAG=runA,... train-quick.slurm
sbatch --export=ALL,RUN_TAG=runB,... train-quick.slurm
sbatch --export=ALL,RUN_TAG=runC,... train-quick.slurm

capture job id <jobid>
--
grep -E "BEST_MODEL_PATH|CHECKPOINT_PATH|Early stopping|Final Test MSE" slurm-<jobid>.out | tail -n 10
grep -E "Epoch|Final Test MSE|Early stopping" slurm-<jobid>.out | tail -n 20
--
sbatch --export=ALL,RUN_TAG=final_best,RESUME=0,NUM_LAYERS=2,HIDDEN_CHANNELS=320,DROPOUT=0.05,LR=2e-4,NUM_EPOCHS=120,LR_PATIENCE=8,EARLY_STOP_PATIENCE=20 train.slurm

--
cp -n best_charge_gcn_final_best.pt model_final_best.pt
cp -n last_charge_gcn_final_best.pt model_final_best_resume.pt
-- safe resume training template --
sbatch --export=ALL,\
RUN_TAG=<new_tag>,\
RESUME=1,\
NUM_LAYERS=2,\
HIDDEN_CHANNELS=320,\
DROPOUT=0.05,\
LR=<new_lr>,\
NUM_EPOCHS=<total_epochs>,\
LR_PATIENCE=12,\
EARLY_STOP_PATIENCE=30,\
CHECKPOINT_PATH=<matching_last_checkpoint.pt>,\
BEST_MODEL_PATH=<new_best_output.pt> \
train.slurm