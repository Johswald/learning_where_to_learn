import os
import pickle
from os import path
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import models
import parser
from utils import utils

class MAML(object):
    def __init__(self, model, inner_loop_params, optimizer_theta=None,
                 optimizer_mask=None, 
                 loss_function=F.cross_entropy, args=None):

        self.model = model
        self.args = args
        self.optimizer_theta = optimizer_theta
        self.optimizer_mask = optimizer_mask
        self.loss_function = loss_function
        self.inner_loop_params = inner_loop_params

    def accuracy(self,logits,targets):
        with torch.no_grad():
            _, predictions=torch.max(logits, dim=1)
            accuracy=torch.mean(predictions.eq(targets).float())
        return accuracy.item()

    def step(self, batch, evaluation=False):
        outer_loss, outer_accuracy, counter=0., 0., 0.
        mask = None

        for _, task in enumerate(zip(*batch["train"], *batch["test"])):
            
            counter += 1
            #DATA
            train_inputs = task[0].to(args.device)
            train_targets = task[1].to(args.device) 
            test_inputs = task[2].to(args.device)
            test_targets = task[3].to(args.device)

            #MASK
            if args.gradient_mask or args.weight_mask:
                mask = self.mask.forward()

            params = OrderedDict()
            for (name, param) in self.model.named_parameters():
                if args.weight_mask and name in mask:
                    params[name] = param*mask[name]
                else:
                    params[name] = param
            #INNER LOOP
            self.model.zero_grad()
            self.optimizer_theta.zero_grad()
            if args.gradient_mask or args.weight_mask:
                self.mask.zero_grad()
                self.optimizer_mask.zero_grad()
                        
            ra = args.gradient_step_sampling
            gs_sample = np.random.randint(low=args.gradient_steps-ra, 
                                          high=args.gradient_steps+ra+1)

            for t in range(gs_sample):
                train_logits = self.model(train_inputs, params=params)
                inner_loss = self.loss_function(train_logits, train_targets)
                self.model.zero_grad()
                grads=torch.autograd.grad(inner_loss, params.values(), 
                                          create_graph=args.second_order)
                params_next=OrderedDict()
                for (name, param), grad in zip(list(params.items()), grads):
                    if args.gradient_mask and name in mask and \
                                              name in self.inner_loop_params:
                        if args.meta_relu_through or args.meta_sgd_linear or \
                           args.meta_relu or args.meta_exp:
                            params_next[name] = \
                                    param-(grad*mask[name])
                        else:
                            params_next[name] = \
                                    param-args.step_size*(grad*mask[name])
                    elif args.weight_mask and name in mask and \
                                              name in self.inner_loop_params:
                        params_next[name] = \
                                    (param-args.step_size*grad)*mask[name]
                    elif name in self.inner_loop_params:
                        params_next[name] = param-args.step_size*grad
                    else:
                        # No inner loop adaptation 
                        params_next[name] = param
                params=params_next
            
            test_logit = self.model(test_inputs, params=params)
            outer_loss += self.loss_function(test_logit, test_targets)
            outer_accuracy += self.accuracy(test_logit, test_targets)

        outer_accuracy = float(outer_accuracy)/counter

        if evaluation:
            # We assume that the test_batch_size is set to 1
            return outer_accuracy

        # crazy zero gradding
        self.optimizer_theta.zero_grad()
        self.model.zero_grad()
        if args.gradient_mask or args.weight_mask:
            self.mask.zero_grad()
            self.optimizer_mask.zero_grad()
        
        # backward and step
        outer_loss.backward()
        self.optimizer_theta.step()
        if args.gradient_mask or args.weight_mask:
            self.optimizer_mask.step()

        return outer_loss.detach(), outer_accuracy, mask

    def train(self, dataloader, max_batches=500, epoch= None):
        
        # Training for one epoch  
        num_batches = 0
        for batch in dataloader:
            if num_batches >= max_batches:
                break
            loss, acc, masks = self.step(batch)
            num_batches += 1
        
        # Write some stats
        with torch.no_grad():
            if args.tensorboard:
                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar('Training Accuracy', acc, epoch)
                if masks is not None:
                    mean_sparsity_z, mean_sparsity_n = 0., 0.
                    for k in masks:
                        cur_sparsity_z = np.count_nonzero(masks[k].\
                                            detach().cpu().numpy())
                        cur_sparsity_n = np.prod(masks[k].shape)

                        mean_sparsity_z += cur_sparsity_z
                        mean_sparsity_n += cur_sparsity_n

                        proc_ones = cur_sparsity_z/cur_sparsity_n
                        writer.add_scalar('zeros (%) in group ' + k, 
                        100 - proc_ones*100, epoch)

                    mean_sparsity = 1 - mean_sparsity_z/mean_sparsity_n
                    writer.add_scalar('mean zeros (%) ' + k, 
                                        mean_sparsity*100, epoch)

        if epoch % args.val_after == 0 or args.epochs - 1 == epoch:
            print("\nTraining epochs ", epoch)
            print("Loss: {:.2f}".format(loss.item()))
            print("Accuracy: {:.2f} %".format(acc*100))
            if masks is not None:
                print("Mean sparsity: {:.2f} % ".format(mean_sparsity*100))
        
        return mean_sparsity

    def evaluate(self, dataloader, num_batches, epoch, test=False,):

        acc, count= 0., 0.
        # evalutating of num_batches / tasks
        for batch in dataloader:
            if count >= num_batches: break
            acc += self.step(batch, evaluation=True)
            count+=1
        acc=acc/count

        # some logging          
        if args.tensorboard:
            if test:
                writer.add_scalar('Test Accuracy',acc, epoch)
            else:
                writer.add_scalar('Val Accuracy',acc, epoch)
        
        print("Accuracy: {:.2f} % \n".format(acc*100))
        return acc

def training_loop(args, metalearner, meta_dataloader, ):

    best_acc_val = 0
    args.best_acc_epoch = args.epochs
    args.best_acc = 0.
    args.mean_sparsity_best = 0.
    ever_checkpointed = False

    for epoch in range(args.epochs):
        # train one epoch
        ms =metalearner.train(meta_dataloader["train"],args.batches_train,epoch)
        # validate performance 
        if epoch % args.val_after == 0 and epoch > args.val_start or \
                                                       args.epochs - 1 == epoch:
            print("---------------------Val---------------------")
            print("Current epoch: ", epoch)
            acc  = metalearner.evaluate(meta_dataloader["val"], 
                                                    args.batches_val, epoch)
            # test if better performance is found
            if best_acc_val < acc and epoch > args.test_start:
                best_acc_val = acc
                print("---------------------Test---------------------")
                acc = metalearner.evaluate(meta_dataloader["test"],
                                            args.batches_test, epoch, test=True)
                args.best_acc_epoch = epoch
                args.best_acc = acc
                args.mean_sparsity_best = ms
                # Checkpoint all networks 
                if args.checkpoint_models:
                    ever_checkpointed = True
                    print("Saving classifier in : ", os.path.join(args.out_dir, 
                                                            'classifier.pth'))
                    torch.save(classifier.state_dict(), os.path.join(
                                                args.out_dir, 'classifier.pth'))
                    if args.gradient_mask or args.weight_mask:
                        print("Saving masking parameters.")
                        torch.save(mask.state_dict(), 
                                    os.path.join(args.out_dir, 'mask.pth'))
                    if args.gradient_mask_plus:
                        print("Saving mask projection parameters.")
                        torch.save(mask.mask_plus.state_dict(), 
                             os.path.join(args.out_dir, 'mask_plus.pth'))

    args.end_acc = acc   
    args.mean_sparsity_end = ms

    print("\nBest checkpointed model:")
    print("Training epoch ", args.best_acc_epoch)
    print("Accuracy: {:.2f} % ".format(args.best_acc*100))
    print("Mean sparsity: {:.2f} % ".format(args.mean_sparsity_best*100))
    
    # CROSS VALIDATION ACC
    if args.checkpoint_models and ever_checkpointed:
        print("\nLoading checkpointed models. ")
        
        if args.checkpoint_models:
            metalearner.model.load_state_dict(torch.load(os.path.join(
                                        args.out_dir, 'classifier.pth')))
            if args.gradient_mask:
                metalearner.mask.load_state_dict(torch.load(
                                    os.path.join(args.out_dir, 
                                    'mask.pth')))
            if args.gradient_mask_plus:
                metalearner.mask.mask_plus.load_state_dict(
                                    torch.load(os.path.join(args.out_dir, 
                                    'mask_plus.pth')))

    acc_cross_datasets=[]
    datasets=["TieredImagenet", "CUB", "CARS"]
    for name in datasets:
        print("Cross-dataset testing on ", name)
        
        args.dataset= name
        meta_dataloader, _,_ = utils.load_data(args)
        acc =metalearner.evaluate(meta_dataloader["test"],
                                    args.batches_test, epoch, test=True)
        
    writer.close()

if __name__ == "__main__":

    """
    **************************************************************
    Main function to train sparse-MAML on few-shot learning tasks. 
    **************************************************************
    """

    parser = parser.get_parser()  
    args = parser.parse_args()

    print("\nFew-shot experiment: {shot}-shot {way}-way on {dataset}".\
                                format(shot=args.num_shots_train, 
                                way=args.num_ways, dataset=args.dataset))
                        
    # SETUP
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

    # TENSORBOARD
    if args.tensorboard:
        writer=SummaryWriter()
        if args.out_dir == "out_dir_default":
            writer=SummaryWriter()
            args.out_dir = writer.log_dir
            with open(os.path.join(args.out_dir, 'config.pickle'), 'wb') as f:
                pickle.dump(args, f)
        else:
            writer=SummaryWriter(args.out_dir)

    # DATA
    meta_dataloader, feature_size, input_channels = utils.load_data(args)


    # META-MODEL
    if not args.resnet:
        classifier = models.MetaConvModel(input_channels,
                                    out_features=args.num_ways,
                                    hidden_size=args.hidden_size,
                                    feature_size=feature_size,
                                    bias=args.bias).to(args.device)
    else:
        classifier = models.ResNet(out_features=args.num_ways, 
                                    big_network=args.big_resnet).to(args.device)
    
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)
    parameters = [{'params': classifier.parameters(), 'lr': args.lr},]

    inner_loop_params = []
    for name, params in classifier.named_parameters():
        num_params =+ np.prod(params.shape)
        if args.resnet:
            # Downsample has batchnorm params named "downsample.1"
            if args.no_bn_in_inner_loop and \
                                ("bn" in name or "downsample.1" in name):
                continue
            if args.no_downsample_in_inner_loop and "downsample" in name:
                continue
        else:
            if args.no_bn_in_inner_loop and "norm" in name:
                continue
        inner_loop_params.append(name)
    # MASK plus
    if args.gradient_mask_plus and (args.gradient_mask or args.weight_mask):
        if args.resnet:
            print("sparse-MAML plus not supported with ResNet12.")
            exit()
        mask_names = []
        for name,params in classifier.named_parameters():
            if "conv.weight" in name and name in inner_loop_params:
                mask_names.append(name)
        #the number of channels for each layer, in order
        mask_channels = [3,64,64,64]
        mask_plus = models.GradientMaskPlus(args,
                                    layer_names=mask_names,
                                    layer_sizes=mask_channels,
                                    feature_size=feature_size).cuda()
    else:
        mask_plus = None

    # MASK
    if args.gradient_mask or args.weight_mask:
        mask_names = []
        shapes = []
        for name, params in classifier.named_parameters():
            if name in inner_loop_params:
                mask_names.append(name)
                shapes.append(params.shape)

        mask = models.GradientMask(args, weight_names= mask_names,
                                        weight_shapes=shapes, 
                                        mask_plus=mask_plus
                                        ).to(args.device)

        # NOTE: The parameters of mask_plus are contained in mask
        if args.optimizer_mask == "ADAM" or args.optimizer_mask == "Adam":
            optimizer_mask=torch.optim.Adam(mask.parameters(),args.mask_lr)

        else:
            optimizer_mask=torch.optim.SGD(mask.parameters(), args.mask_lr, 
                                                momentum=args.momentum,
                                                nesterov=(args.momentum > 0.0))
        print("\nMask optimizer:", optimizer_mask)

    else:
        optimizer_mask = None

    # OPTIMIZER
    if args.optimizer_theta == "ADAM" or args.optimizer_theta == "Adam":
        optimizer_theta=torch.optim.Adam(parameters, args.lr)

    else:
        optimizer_theta=torch.optim.SGD(parameters, args.lr, 
                                                momentum=args.momentum,
                                                nesterov=(args.momentum > 0.0))
    print("\nTheta optimizer:", optimizer_theta)

    # MAML object
    metalearner=MAML(classifier, optimizer_theta=optimizer_theta,
                    optimizer_mask=optimizer_mask, 
                    loss_function=loss_function, 
                    inner_loop_params=inner_loop_params,
                    args=args)

    if args.gradient_mask or args.weight_mask:
        metalearner.mask=mask
        if args.gradient_mask_plus:
            metalearner.mask=mask

    training_loop(args, metalearner, meta_dataloader)
