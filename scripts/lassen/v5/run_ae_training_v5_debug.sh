source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/g/g91/cutforth1/mf-ae
cd /g/g91/cutforth1/mf-ae
conda run -n genmodel_env accelerate launch ./src/main/main_train.py --debug  --dataset-type ellipse --num-dl-workers 0 --batch-size 1 --num-epochs 10 --save-and-sample-every 1 --lr 0.0001 --loss l1 --dim-mults 1 2 4 8 8 --block-type 1 --z-channels 4 --dim 4 --interface-representation heaviside --run-name run_debug_v5