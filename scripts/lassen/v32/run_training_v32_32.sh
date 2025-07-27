source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py  --seed 4 --model-type baseline --dim 32 --dataset-type volumetric --num-dl-workers 0 --batch-size 1 --num-epochs 15 --lr 1e-05 --loss mse --dim-mults 1 2 4 8 8 8 --block-type 1 --z-channels 4 --max-samples 25000 --max-train-samples 10000 --data-dir /usr/workspace/cutforth1/data-mf-ae/patched_hit_experiment/SIGNED_DISTANCE_EXACT --weight-decay 0.0001 --act-type tanh --run-name interfacial_ae_v32_run_32_datadirSIGNED_DISTANCE_EXACT_lossmse_lr1e05_weightdecay00001_acttypetanh
