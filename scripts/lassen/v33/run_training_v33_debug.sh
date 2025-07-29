source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py --debug  --act-type silu --lr 1e-05 --model-type baseline --dim 32 --dataset-type volumetric --num-dl-workers 0 --batch-size 1 --num-epochs 15 --loss auto --dim-mults 1 2 4 8 8 8 --block-type 1 --z-channels 4 --max-samples 25000 --max-train-samples 10000 --data-dir /usr/workspace/cutforth1/data-mf-ae/patched_hit_experiment/SIGNED_DISTANCE_EXACT --seed 0 --run-name run_debug_v33
