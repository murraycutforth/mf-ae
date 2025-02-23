source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py  --model-type conv_with_fc --dim 32 --dataset-type volumetric --num-dl-workers 0 --batch-size 1 --num-epochs 5 --lr 0.0001 --loss l1 --dim-mults 1 2 4 8 8 4 --block-type 1 --z-channels 4 --max-samples 25000 --data-dir /usr/workspace/cutforth1/data-mf-ae/patched_hit_experiment/TANH_EPSILON0.125 --seed 4 --fc-layers 4096 512 64 --run-name interfacial_ae_v26_run_19_datadirTANH_EPSILON0125_seed4_fclayers409651264
