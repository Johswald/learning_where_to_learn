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

# 5-SHOT RESNET
exe echo training 5-SHOT sparse-ReLU-MAML RESNET
python3 train.py --epochs=40 --batches_train=1000 --batches_test=500 --batches_val=500 --batch_size=1  --num_shots_train=5 --gradient_mask --step_size=0.05 --mask_lr=0.01 --checkpoint_models --gradient_steps=35 --val_start=10 --test_start=10 --tensorboard --val_after 1 --resnet --no_bn_in_inner_loop --meta_relu_through --meta_constant_init --clamp_outer_gradients
# 5-SHOT RESNET
exe echo training 5-SHOT sparse-MAML RESNET
python3 train.py --epochs=40 --batches_train=1000 --batches_test=500 --batches_val=500 --batch_size=1  --num_shots_train=5 --gradient_mask --step_size=0.05 --mask_lr=0.01 --checkpoint_models --gradient_steps=35 --val_start=10 --test_start=10 --tensorboard --val_after 1 --resnet --no_bn_in_inner_loop --clamp_outer_gradients

# 1-SHOT RESNET
exe echo training 1-SHOT sparse-ReLU-MAML RESNET
python3 train.py --epochs=40 --batches_train=1000 --batches_test=500 --batches_val=500 --batch_size=1  --num_shots_train=1 --gradient_mask --step_size=0.05 --mask_lr=0.01 --checkpoint_models --gradient_steps=35 --val_start=10 --test_start=10 --tensorboard --val_after 1 --resnet --no_bn_in_inner_loop --meta_relu_through --meta_constant_init --clamp_outer_gradients

# 1-SHOT RESNET
exe echo training 1-SHOT sparse-MAML RESNET
python3 train.py --epochs=40 --batches_train=1000 --batches_test=500 --batches_val=500 --batch_size=1  --num_shots_train=1 --gradient_mask --step_size=0.05 --mask_lr=0.01 --checkpoint_models --gradient_steps=35 --val_start=10 --test_start=10 --tensorboard --val_after 1 --resnet --no_bn_in_inner_loop --clamp_outer_gradients

