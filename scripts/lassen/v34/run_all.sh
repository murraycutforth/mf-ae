#!/bin/bash
bsub run_training_v34_0.bsub
sleep 0.5
bsub run_training_v34_1.bsub
sleep 0.5
bsub run_training_v34_2.bsub
sleep 0.5
bsub run_training_v34_3.bsub
sleep 0.5
bsub run_training_v34_4.bsub
sleep 0.5
bsub run_training_v34_5.bsub
sleep 0.5
bsub run_training_v34_6.bsub
sleep 0.5
bsub run_training_v34_7.bsub
sleep 0.5
bsub run_training_v34_8.bsub
sleep 0.5
echo "All jobs submitted"