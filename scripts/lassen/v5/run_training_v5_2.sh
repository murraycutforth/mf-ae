source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py  --dim 32 --dataset-type ellipse --vol-size 64 --num-dl-workers 0 --batch-size 1 --num-epochs 75 --save-and-sample-every 75 --lr 0.0001 --loss l1 --dim-mults 1 2 4 8 8 --block-type 1 --z-channels 4 --data-dir /usr/WS1/cutforth1/data-mf-ae/v5_ellipsoids/HEAVISIDE --seed 0 --run-name interfacial_ae_v5_run_02_datadir/usr/WS1/cutforth1/datamfae/v5_ellipsoids/HEAVISIDE_seed0
