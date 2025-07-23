source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py  --model-type baseline --dim 32 --dataset-type volumetric --num-dl-workers 0 --batch-size 1 --num-epochs 100 --lr 0.0001 --loss l1 --dim-mults 1 2 4 8 8 8 --block-type 1 --z-channels 4 --max-samples 25000 --data-dir /usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.015625 --seed 4 --max-train-samples 2000 --run-name interfacial_ae_v31_run_139_datadir/usr/workspace/cutforth1/datamfae/spheres_mu_250/TANH_EPSILON0015625_seed4_maxtrainsamples2000
