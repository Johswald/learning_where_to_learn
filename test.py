import os
import pickle
from os import path
import numpy as np
import collections
import itertools
from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

import models
from utils import data_utils as utils

class MAML(object):
    def __init__(self, model, attention=None, optimizer_theta=None,
                 optimizer_mask=None, loss_function=F.cross_entropy, args=None):

        self.model = model
        self.attention = attention
        self.args = args
        self.optimizer_theta = optimizer_theta
        self.optimizer_mask = optimizer_mask
        self.loss_function = loss_function

    def accuracy(self,logits,targets):
        with torch.no_grad():
            _, predictions=torch.max(logits, dim=1)
            accuracy=torch.mean(predictions.eq(targets).float())
        return accuracy.item()

    def step(self, batch):
        outer_loss, outer_accuracy, counter=0., 0., 0.
        masks = None

        for task_id, task in enumerate(zip(*batch["train"], *batch["test"])):
            
            counter += 1
            
            #DATA
            train_inputs = task[0].to(args.device)
            train_targets = task[1].to(args.device) 
            test_inputs = task[2].to(args.device)
            test_targets = task[3].to(args.device)

            #MASK
            if args.gradient_attention or args.weight_attention:
                masks = self.attention.forward(train_inputs)

            params = OrderedDict()
            for (name, param) in self.model.named_parameters():
                if args.weight_attention and name in masks:
                    params[name] = param*masks[name]
                else:
                    params[name] = param
            
            #INNER LOOP
            
            # crazy zero gradding
            self.model.zero_grad()
            self.optimizer_theta.zero_grad()
            if args.gradient_attention or args.weight_attention:
                self.attention.zero_grad()
            if args.gradient_attention or args.weight_attention:
                self.optimizer_mask.zero_grad()
            
            
            ra = args.gradient_step_sampling
            gs_sample = np.random.randint(low=args.gradient_steps-ra, 
                                          high=args.gradient_steps+ra+1)

            for t in range(gs_sample):
                train_logits = self.model(train_inputs, params=params)
                inner_loss = self.loss_function(train_logits,train_targets)
                self.model.zero_grad()
                grads=torch.autograd.grad(inner_loss, params.values(), 
                                          retain_graph=args.second_order,
                                          create_graph=args.second_order)
                if args.dynamic_mask:
                    masks = self.attention.forward(train_inputs, t=t+1)
                params_next=OrderedDict()
                for (name, param), grad in zip(list(params.items()), grads):
                    if args.gradient_attention and name in masks:
                        params_next[name] = \
                                    param-args.step_size*(grad*masks[name])
                    elif args.weight_attention and name in masks:
                        params_next[name] = \
                                    (param-args.step_size*grad)*masks[name]
                    else:
                        params_next[name] = param-args.step_size*grad
                params=params_next

            #IOUTER LOOP
            test_logit = self.model(test_inputs, params=params)
            outer_loss += self.loss_function(test_logit,test_targets)
            outer_accuracy += self.accuracy(test_logit,test_targets)

        outer_accuracy = float(outer_accuracy)/counter
        # crazy zero gradding
        self.optimizer_theta.zero_grad()
        self.model.zero_grad()
        if args.gradient_attention or args.weight_attention:
            self.attention.zero_grad()
        if args.gradient_attention or args.weight_attention:
            self.optimizer_mask.zero_grad()
        
        # backward and step
        outer_loss.backward()
        self.optimizer_theta.step()
        if args.gradient_attention or args.weight_attention:
            self.optimizer_mask.step()
        return outer_loss.detach(), outer_accuracy, masks

    def train(self, dataloader, max_batches=500, epoch= None):

        num_batches = 0
        for batch in dataloader:
            if num_batches >= max_batches:
                break
            loss, acc, masks = self.step(batch)
            num_batches += 1

        # write some stats
        with torch.no_grad():
            if args.tensorboard:
                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar('training Accuracy', acc, epoch)
                if masks is not None:
                    mean_sparsity = 0
                    mean_sparsity_z = 0
                    mean_sparsity_n = 0

                    for k in masks:
                        mean_sparsity_z += np.count_nonzero(masks[k].\
                                            detach().cpu().numpy())
                        mean_sparsity_n += np.prod(masks[k].shape)

                        proc_ones = np.count_nonzero(masks[k].\
                                detach().cpu().numpy())/np.prod(masks[k].shape)
                        writer.add_scalar('zeros (%) in group ' + k, 
                        1 - proc_ones, epoch)
                    mean_sparsity = 1 - mean_sparsity_z/mean_sparsity_n
                    writer.add_scalar('zeros (%) overall training' + k, 
                                        mean_sparsity, epoch)

        if epoch % args.val_after == 0 or args.epochs - 1 == epoch:
            print("\nTraining epochs ", epoch)
            print("Training batches: {:}".format(num_batches))
            print("Loss: {:.2f}".format(loss.item()))
            print("Accuracy: {:.2f} %".format(acc*100))
            if masks is not None:
                print("Mean sparsity: {:.2f} % ".format(mean_sparsity*100))

        if args.dynamic_mask:
            self.attention.analyse_dynamic_mask(writer=writer, 
                                                                    t=epoch)

    def step_evaluate(self, batch, epoch, batch_iter, debug_in_data=None):
        outer_loss, outer_accuracy, counter = 0., 0., 0.
        outer_accuracy_list = []

        masks = None
        
        for task_id, task in enumerate(zip(*batch["train"], *batch["test"])):
            counter += 1

            #DATA
            train_inputs = task[0].to(args.device)
            train_targets = task[1].to(args.device) 
            test_inputs = task[2].to(args.device)
            test_targets = task[3].to(args.device)


            #MASK
            with torch.no_grad():
                if args.gradient_attention or args.weight_attention:
                    masks = self.attention.forward(train_inputs)
                
            params = OrderedDict()
            for (name, param) in self.model.named_parameters():
                if args.weight_attention and name in masks:
                    params[name] = param*masks[name]
                else:
                    params[name] = param

            #INNER LOOP 
            for g_step in range(args.gradient_steps_test):
                
                train_logits=self.model(train_inputs, params=params)
                inner_loss=self.loss_function(train_logits, train_targets)

                self.model.zero_grad()
                grads=torch.autograd.grad(inner_loss, params.values())
                params_next=OrderedDict()
                
                with torch.no_grad():
                    if args.dynamic_mask:
                        masks = self.attention.forward(train_inputs, 
                                                        t = g_step+1)

                for (name, param), grad in zip(list(params.items()), grads):
                    if args.gradient_attention and name in masks:
                        params_next[name] = \
                                    param-args.step_size*(grad*masks[name])
                    elif args.weight_attention and name in masks:
                        params_next[name] = \
                                    (param-args.step_size*grad)*masks[name]
                    else:
                        params_next[name] = param-args.step_size*grad
                params=params_next
                
                with torch.no_grad():

                    if g_step % 5 == 0 or g_step == args.gradient_steps_test-1:
                        test_logit=self.model(test_inputs,params=params)
                        outer_accuracy_list.append(self.accuracy(test_logit, 
                                                                 test_targets))
                        
            test_logit=self.model(test_inputs,params=params)
            outer_loss+=self.loss_function(test_logit,test_targets)
            outer_accuracy += self.accuracy(test_logit,test_targets)

        outer_accuracy=float(outer_accuracy)/counter
        outer_accuracy_list = np.array(outer_accuracy_list)/counter
        self.model.zero_grad()
        
        return outer_loss.detach()/counter, outer_accuracy, masks, \
                                              outer_accuracy_list, train_inputs

    def evaluate(self, dataloader, num_batches, epoch=None, test= False,):

        # crazy zero gradding
        self.optimizer_theta.zero_grad()
        self.model.zero_grad()
        if args.gradient_attention or args.weight_attention:
            self.attention.zero_grad()
        if args.gradient_attention or args.weight_attention:
            self.optimizer_mask.zero_grad()

        loss, acc, count= 0.,0.,0.
        oa_list = 0.
        masks_list = []
        
        # evalutating of num_batches / tasks
        for batch in dataloader:
            if count >= num_batches: break
            outer_loss, outer_accuracy, masks, \
                oa_list_c, in_data = self.step_evaluate(batch, epoch, count)

            loss+=outer_loss
            acc+=outer_accuracy
            count+=1
            oa_list += oa_list_c
            # turn of hamming distance computing 
            if args.x_dep_masking and False:
                masks_list.append(masks)

        loss=loss/count
        acc=acc/count
        oa_list = np.array(oa_list)/count
        

        # some logging
                    
        if args.tensorboard:
            if test:
                writer.add_scalar('Test Loss',loss, epoch)
                writer.add_scalar('Test Accuracy',acc, epoch)
            else:
                writer.add_scalar('Val Loss',loss, epoch)
                writer.add_scalar('Val Accuracy',acc, epoch)
        
        print("Loss: {:.4f}".format(loss))
        print("Accuracy: {:.2f} % \n".format(acc*100))
        

        # hamming distance compute (turned off)
        if args.x_dep_masking and False:
            print("computing H distance")
            hamming_mask_all = 0
            hamming_count = 0
            for mask_a, mask_b in itertools.combinations(masks_list, 2):
                hamming_count += 1
                hamming_mask = 0
                for k in mask_a:
                    k_a = mask_a[k].detach().cpu().numpy()
                    k_b = mask_b[k].detach().cpu().numpy()
                    hamming_mask += np.count_nonzero(k_a - k_b)
                hamming_mask_all += hamming_mask
            hamming_avg = hamming_mask_all/hamming_count
            print("Avg hamming distance masks:  {:.2f}\n".format(hamming_avg))
            writer.add_scalar('Avg hamming distance masks', hamming_avg, epoch)

        if masks is not None:
            mean_sparsity_z = 0
            mean_sparssity_n = 0
            for k in masks:
                mean_sparsity_z += \
                               np.count_nonzero(masks[k].detach().cpu().numpy())
                mean_sparssity_n += np.prod(masks[k].shape)
            mean_sparsity = mean_sparsity_z / mean_sparssity_n
        else:
            mean_sparsity = 0
        
        mean_sparsity = 1. - mean_sparsity

        if args.dynamic_mask:
            dyn_mask_sparsity = self.attention.analyse_dynamic_mask()
            print("Average dyn mask sparsity", dyn_mask_sparsity)

        if args.tensorboard:
            if test:
                writer.add_scalar('Test mean sparsity', mean_sparsity, epoch)
            else:
                writer.add_scalar('Val mean sparsity', mean_sparsity, epoch)

        return acc, loss, mean_sparsity, oa_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MamlMask')

    # General
    parser.add_argument('--out_dir', type=str, default="out_dir_default",
            help='Path to the output folder for saving the model.')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Use tensorboard')
    parser.add_argument('--val_after',type=int, default=5, metavar='va',
                        help='classifier hidden size')
    parser.add_argument('--test_start',type=int, default=150, metavar='va',
                        help='classifier hidden size')
    parser.add_argument('--val_start',type=int, default=100, metavar='va',
                        help='classifier hidden size')
    parser.add_argument('--checkpoint_models', action='store_true',
                        help='Checkpoint models')
    parser.add_argument('--dont_use_cuda', action='store_true',
                        help='Dont use CUDA if available.')
    parser.add_argument('--seed', type=int, default=-1, 
                        help='Random seed.')
    parser.add_argument('--set_seed', action='store_true',
                        help='Set random seed.')

    # Training hps
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Meta-training batch size (#tasks)')
    parser.add_argument('--test_batch_size', type=int, default=12, 
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
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for dataloader.')

    # Main network config
    parser.add_argument('--hidden_size',type=int,default=64, 
                        help='Classifier hidden size')

    # Few-shot experiment config
    parser.add_argument('--num_shots_train',type=int,default=5, 
                        help='K shots training phase')
    parser.add_argument('--num_shots_test',type=int,default=15, 
                    help='K shots test phase')
    parser.add_argument('--num_ways',type=int,default=5, 
                        help='Number of labels per task')
    parser.add_argument('--dataset', type=str, default="MiniImagenet",
                        help='Datsets supported: MiniImageNet, Omniglot')            
    parser.add_argument('--mini_crop', action='store_true',default=False,
                        help='Data augment crop (as in Meta-Curvature?)')

    # MAML hyperparameters
    parser.add_argument('--step_size',type=float,default=0.1, 
                        help='Inner loop learning rate.')
    parser.add_argument('--gradient_steps',type=int,default=35,metavar='gs',
                        help='Number of gradient steps in inner loop')
    parser.add_argument('--gradient_step_sampling',type=int, default=5,
                        help='Range of gradient steps sampled.')
    parser.add_argument('--gradient_steps_test',type=int,default=100,
                        help='Number of gradient steps in inner loop')
    parser.add_argument('--second_order',action='store_true',default=False,
                        help='Use second order in MAML')
    

    #masking hyperparameters
    parser.add_argument('--gradient_attention', action='store_true',
                        help='mask the gradient')
    parser.add_argument('--weight_attention', action='store_true',default=False,
                        help='Mask the weights')
    parser.add_argument('--x_dep_masking',action='store_true', default=False,
                        help='mask the gradient with x dep network')
    parser.add_argument('--x_output_shift',type=float, default=0.,
                        help='Shift the output of the x dep gating to ' +
                            'enforce higher / lower sparsity in the beginning.')
    parser.add_argument('--weight_noise',type=float, default=0.,
                        help='Noise perturbing weights underlying the mask.')

    parser.add_argument('--no_bn_masking',action='store_true', default=False,
                        help='No masking of the BatchNorm weights')
    parser.add_argument('--no_head_masking',action='store_true', default=False,
                        help='No masking of the head')  
    parser.add_argument('--init_shift', type=float, default=0.0, 
                        help='Shift mask init.')
    parser.add_argument('--kaiming_init', action='store_true',default=False,
                        help='Mask init.')
    parser.add_argument('--meta_relu_sgd', action='store_true',default=False,
                        help='Meta-SGD with relu instead of masking.')
    parser.add_argument('--meta_relu_through',action='store_true',default=False,
                        help='Meta-SGD straigth through relu.')
    parser.add_argument('--meta_sgd_linear', action='store_true',default=False,
                        help='Meta-SGD linear instead of Relu.') 
    parser.add_argument('--meta_sgd_init', action='store_true',
                        help='Init uniform [0.005, 0.1].')
    parser.add_argument('--meta_exp', action='store_true',
                                    help='Init uniform [0.005, 0.1].')
    #ime dynamic mask
    parser.add_argument('--dynamic_mask', action='store_true', 
                        help='Use the same x dep input to debug.')

    # X debug
    parser.add_argument('--x_debug', action='store_true',
                        help='Investigate what happens if one cuts the x dep.')
    parser.add_argument('--x_debug_noise',  type=float, default=0.0, 
                        help='Noise of the fake x dep. embedding')
                    

    args = parser.parse_args()

    if args.set_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.out_dir_w = args.out_dir
    if args.out_dir != "out_dir_default":
        if not path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            print("Created output folder %s." % (args.out_dir))
        # Save user configs to ensure reproducibility of this experiment.
        with open(os.path.join(args.out_dir, 'config.pickle'), 'wb') as f:
            pickle.dump(args, f)

    args.device = torch.device('cuda' if not args.dont_use_cuda
                                and torch.cuda.is_available() else 'cpu')

    #TENSORBOARD
    if args.tensorboard:
        #writer=SummaryWriter("backattention/{}".format(args.name))
        writer=SummaryWriter()
        if args.out_dir == "out_dir_default":
            writer=SummaryWriter()
            args.out_dir = writer.log_dir
            with open(os.path.join(args.out_dir, 'config.pickle'), 'wb') as f:
                pickle.dump(args, f)
        else:
            writer=SummaryWriter(args.out_dir)

    #DATA
    meta_dataloader, feature_size, input_channels = utils.load_data(args)


    #META-MODEL
    classifier = models.MetaConvModel(input_channels,out_features=args.num_ways,
                                    hidden_size=args.hidden_size,
                                    feature_size=feature_size).to(args.device)

    loss_function=torch.nn.CrossEntropyLoss().to(args.device)
    parameters=[{'params': classifier.parameters(), 'lr': args.lr},]

    # X-dep mask
    if args.x_dep_masking and (args.gradient_attention or args.weight_attention):
        attention_names=[]
        for name,params in classifier.named_parameters():
            if "conv.weight" in name:
                attention_names.append(name)

        #the number of channels for each layer, in order
        attention_channels=[3,64,64,64]

        x_conv_attention=models.XAttentionMask(layer_names=attention_names,
                                    layer_sizes=attention_channels,
                                    input_channels=3, 
                                    hidden_size=args.hidden_size, 
                                    meta_sgd_linear = args.meta_sgd_linear,
                                    meta_relu_sgd=args.meta_relu_sgd,
                                    meta_relu_through=args.meta_relu_through,
                                    feature_size=feature_size,
                                    x_debug =args.x_debug,
                                    x_debug_noise=args.x_debug_noise,
                                    out_shift=args.x_output_shift).cuda()

    else:
        x_conv_attention = None

    # MASK
    if args.gradient_attention or args.weight_attention:
        attention_names = []
        shapes = []
        for name, params in classifier.named_parameters():
            attention_names.append(name)
            shapes.append(params.shape)

        attention = models.AttentionMask(weight_names= attention_names, 
                                    weight_shapes=shapes, 
                                    kaiming_init=args.kaiming_init,
                                    init_shift=args.init_shift,
                                    x_conv_attention=x_conv_attention,
                                    noise_std = args.weight_noise,
                                    no_bn_masking=args.no_bn_masking,
                                    no_head_masking=args.no_head_masking,
                                    meta_sgd_linear = args.meta_sgd_linear,
                                    meta_sgd_init =args.meta_sgd_init,
                                    meta_exp =args.meta_exp,
                                    meta_relu_sgd=args.meta_relu_sgd,
                                    meta_relu_through=args.meta_relu_through,
                                    alpha_init=args.step_size,
                                    dynamic_mask = args.dynamic_mask
                                    ).to(args.device)

        # NOTE: The parameters of x_conv_attention are contained in attention
        
        if args.optimizer_mask == "ADAM" or args.optimizer_mask == "Adam":
            optimizer_mask=torch.optim.Adam(attention.parameters(),args.mask_lr)

        else:
            optimizer_mask=torch.optim.SGD(attention.parameters(), args.mask_lr, 
                                                momentum=args.momentum,
                                                nesterov=(args.momentum > 0.0))
        print("\nMask optimizer", optimizer_mask)

    else:
        optimizer_mask=None

    # OPTIMIZER
    if args.optimizer_theta == "ADAM" or args.optimizer_theta == "Adam":
        optimizer_theta=torch.optim.Adam(parameters, args.lr)

    else:
        optimizer_theta=torch.optim.SGD(parameters, args.lr, 
                                                momentum=args.momentum,
                                                nesterov=(args.momentum > 0.0))
    print("\nTheta optimizer", optimizer_theta)

    # MAML object
    metalearner=MAML(classifier, optimizer_theta=optimizer_theta,
                    optimizer_mask=optimizer_mask, 
                    loss_function=loss_function, 
                    args=args)

    if args.gradient_attention or args.weight_attention:
        metalearner.attention=attention

    # TRAINING LOOP
    metalearner.model.load_state_dict(torch.load(os.path.join(
                                        args.out_dir, 'classifier.pth')))
    if args.gradient_attention:
        metalearner.attention.load_state_dict(torch.load(
                                    os.path.join(args.out_dir, 
                                    'attention.pth')))
            
    if args.x_dep_masking:
        metalearner.attention.x_conv_attention.load_state_dict(
                                    torch.load(os.path.join(args.out_dir, 
                                    'x_conv_attention.pth')))
    
    for name, m in classifier.named_parameters():
        print(name)
    import matplotlib.pyplot as plt
    for name, m in attention.named_parameters():
        np_m = m.detach().cpu().numpy().flatten()
        histo = np.histogram(np_m)
        fig = plt.figure()
        _ = plt.hist(np_m, bins='auto')
        fig.savefig(name + '.png')
        print(name, torch.mean(m), torch.std(m))
    print("loaded")
    

