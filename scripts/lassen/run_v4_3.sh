source /usr/workspace/cutforth1/anaconda/bin/activate
export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib
export PYTHONPATH=/usr/workspace/cutforth1/density-generative-model

cd /g/g91/cutforth1/mf-ae

conda run -n genmodel_env python -m src.main.main_train --data-dir="/p/vast1/cutforth1/mf-ae-data" --outdir="/p/vast1/cutforth1/mf-ae-output" --run-name="v4_3" --batch-size="1" --num-epochs="100" --lr="1e-4" --feat-map-sizes 8 16 32 64 4 --linear-layer-sizes 2000 1000 800 --activation="elu" --normalization="instance" --l2_reg="1e-6" --loss="mse"

