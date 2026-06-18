set -e

cd /pscratch/sd/s/skw/CDS_SHARED/mw-mesh-420s/files/acoustic-elastic-coupled-mw
num_procs=$1

srun -n 16 -u python ~/AutonomousOED/oed_hist.py   --h5_store_K ../K_prior_4_2d.h5   --checkpoint_file large_samps.txt  --r_sq 1e4 --no_plot --budget 175 --total_samples 10000