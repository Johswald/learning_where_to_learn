import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='sparse MAML')

    # General
    parser.add_argument('--out_dir', type=str, default="out_dir_default",
                        help='Path to the output folder for saving the models.')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Use tensorboard')
    parser.add_argument('--val_after',type=int, default=5, metavar='va',
                        help='Start computing validation set acc/checkpointing')
    parser.add_argument('--test_start',type=int, default=-1, metavar='va',
                        help='classifier hidden size')
    parser.add_argument('--val_start',type=int, default=-1, metavar='va',
                        help='classifier hidden size')
    parser.add_argument('--checkpoint_models', action='store_true',
                        help='Checkpoint models')
    parser.add_argument('--dont_use_cuda', action='store_true',
                        help='Dont use CUDA if available.')
    parser.add_argument('--seed', type=int, default=-1, 
                        help='Random seed needed for hpsearch - has no effect.')

    # Training hps
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Meta-training batch size (#tasks)')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                        help='Meta-test batch size (#tasks)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of epochs to train')
    parser.add_argument('--batches_train', type=int, default=100, 
                        help='Number of meta-training task per epoch')
    parser.add_argument('--batches_test',type=int,default=300, metavar='btt',
                        help='Number of meta-testing tasks')
    parser.add_argument('--batches_val',type=int,default=300,
                        help='Number of meta-testing tasks if validating')

    # Optimizer config
    parser.add_argument('--optimizer_theta',type=str,default="ADAM",
                        help='Outer loop optimizer.Use Adam or SGD.')
    parser.add_argument('--optimizer_mask',type=str,default="ADAM",
                        help='Outer loop optimizer for masks.Use Adam or SGD.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Outer loop learning rate.')
    parser.add_argument('--mask_lr',type=float, default=0.001,
                        help='Outer loop learning rate for masks.')
    parser.add_argument('--momentum',type=float,default=0.9, 
                        help='Momentum for SGD (turns on Nesterov)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for dataloader.')

    # Main network config
    parser.add_argument('--resnet', action='store_true',
                        help='Use Resnet12 as classifier.')
    parser.add_argument('--big_resnet', action='store_true',
                        help='Use big Resnet12 as classifier.') 
    parser.add_argument('--hidden_size',type=int,default=64, 
                        help='Classifier hidden size when using ConvNet')
    parser.add_argument('--bias', action='store_true',
                        help='Bias parameters in the network or not')

    # Few-shot experiment config
    parser.add_argument('--num_shots_train',type=int,default=5, 
                        help='K shots training phase')
    parser.add_argument('--num_shots_test',type=int,default=15, 
                    help='K shots test phase')
    parser.add_argument('--num_ways',type=int,default=5, 
                        help='Number of labels per task')
    parser.add_argument('--dataset', type=str, default="MiniImagenet",
                        help='Datsets supported: MiniImageNet, Omniglot')            
    parser.add_argument('--data_aug', action='store_true',
                        help='Augment MiniImageNet data')
    # MAML hyperparameters
    parser.add_argument('--step_size',type=float,default=0.1, 
                        help='Inner loop learning rate.')
    parser.add_argument('--gradient_steps',type=int,default=35,metavar='gs',
                        help='Number of gradient steps in inner loop')
    parser.add_argument('--gradient_step_sampling',type=int, default=0,
                        help='Range of gradient steps sampled.')
    parser.add_argument('--gradient_steps_test',type=int,default=100,
                        help='Number of gradient steps in inner loop')
    parser.add_argument('--second_order',action='store_true',default=False,
                        help='Use second order in MAML (NOT TESTED)')
    parser.add_argument('--no_bn_in_inner_loop',action='store_true',
                        help='Dont adapt BatchNorm params in inner loop')
    parser.add_argument('--no_downsample_in_inner_loop',action='store_true',
                        help='Dont adapt Downsample (ResNet) par in inner loop')

    #masking hyperparameters
    parser.add_argument('--gradient_mask', action='store_true',default=False,
                        help='Mask the gradient.')
    parser.add_argument('--weight_mask', action='store_true',default=False,
                        help='Mask the weights.')
    parser.add_argument('--gradient_mask_plus',action='store_true',
                        help='Mask the gradient with noisy mask')
    parser.add_argument('--plus_output_shift',type=float, default=0.,
                        help='Shift the output gradient mask plus to ' +
                            'enforce higher / lower sparsity in the beginning.')
    parser.add_argument('--init_shift', type=float, default=0.0, 
                        help='Shift mask init to control init sparsity.')
    parser.add_argument('--kaiming_init', action='store_true',default=False,
                        help='Mask init.')
    parser.add_argument('--meta_relu', action='store_true',default=False,
                        help='Meta-SGD with relu instead of masking.')
    parser.add_argument('--meta_relu_through',action='store_true',default=False,
                        help='Meta-SGD straigth through relu.')
    parser.add_argument('--meta_sgd_linear', action='store_true',default=False,
                        help='Meta-SGD linear instead of Relu.') 
    parser.add_argument('--meta_sgd_init', action='store_true',
                        help='Init uniform [0.005, 0.1].')
    parser.add_argument('--meta_exp', action='store_true',
                                    help='Init uniform [0.005, 0.1].')

    return parser