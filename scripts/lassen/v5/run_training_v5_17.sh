source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/WS1/cutforth1/mf-ae
cd /usr/WS1/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py  --dataset-type ellipse --vol-size 64 --num-dl-workers 0 --batch-size 1 --num-epochs 30 --save-and-sample-every 5 --lr 0.0001 --loss l1 --dim-mults 1 2 4 8 8 --block-type 1 --z-channels 4 --dim 32 --interface-representation sdf --epsilon 0.03125 --run-name interfacial_ae_v5_run_17_dim32_interfacerepresentationsdf_epsilon003125
