#!/bin/bash

##### sparse La-MAML #####
exe() { echo "\$ $@" ; "$@" ; }

# 5-SHOT 4-layer ConvNet
exe echo training 5-SHOT sparse-MAML 4-layer ConvNet
python3 train.py --epochs=400 --batches_train=100 --batches_test=500 --batches_val=500 --batch_size=2  --num_shots_train=5 --gradient_mask --step_size=0.25 --mask_lr=0.0075 --checkpoint_models --gradient_steps=35 --val_start=100 --test_start=100 --kaiming_init --tensorboard --val_after 5

# 5-SHOT 4-layer ConvNet
exe echo training 5-SHOT sparse-MAML PLUS 4-layer ConvNet
python3 train.py --epochs=600 --batches_train=100 --batches_test=500 --batches_val=500 --batch_size=2  --num_shots_train=5 --gradient_mask --step_size=0.1 --mask_lr=0.0075 --checkpoint_models --gradient_steps=35 --val_start=100 --test_start=100 --kaiming_init --tensorboard --val_after 5 --gradient_mask_plus --optimizer_mask=SGD --optimizer_theta=SGD

# 1-SHOT 4-layer ConvNet
exe echo training 1-SHOT sparse-MAML 4-layer ConvNet
python3 train.py --epochs=400 --batches_train=100 --batches_test=500 --batches_val=500 --batch_size=2  --num_shots_train=5 --gradient_mask --step_size=0.1 --mask_lr=0.0075 --checkpoint_models --gradient_steps=35 --val_start=100 --test_start=100 --kaiming_init --tensorboard --val_after 5

# 1-SHOT 4-layer ConvNet
exe echo training 1-SHOT sparse-MAML PLUS 4-layer ConvNet
python3 train.py --epochs=600 --batches_train=100 --batches_test=500 --batches_val=500 --batch_size=2  --num_shots_train=5 --gradient_mask --step_size=0.1 --mask_lr=0.0075 --checkpoint_models --gradient_steps=35 --val_start=100 --test_start=100 --kaiming_init --tensorboard --val_after 5 --gradient_mask_plus --optimizer_mask=SGD --optimizer_theta=SGD

