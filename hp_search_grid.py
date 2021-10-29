##########################################
### Please define all parameters below ###
##########################################

grid = {
    'checkpoint_models':[True],
    'epochs': [350],
    'batches_train': [100],
    'batches_test': [500],
    'batches_val': [500],
    'batch_size': [2],
    'test_batch_size': [1],
    'num_shots_test': [15],
    'num_shots_train': [1,5],
    'weight_attention': [False], 
    'gradient_attention': [False, True], # turns on the mask
    # turns of masks in the head or batchnorm parameters
    'no_bn_masking': [False],
    'no_head_masking': [False],
     # shifts the sparsity "initialisation", Use values in [-0.5, 0.5]
    'init_shift': [0.], 
     # this will not set the seed but only rerun the experiment with random seeds
    'seed': [1,2,3,4], 
     # use straight through MAML instead of sparse MAML or sparse MAML+
    'meta_relu_through': [False],
    'bias': [True],
    'meta_bn':[True],
    'meta_sgd_init':[False],
    'meta_exp':[False],
    'step_size': [0.5],
    'gradient_steps': [35],
    'mask_lr': [0.005],
    'gradient_step_sampling': [0],
    'x_dep_masking': [False],
    'x_debug': [False],
    'val_after': [5],
    'val_start':[100],
    'test_start':[100],
    'x_debug_noise':[0.],
    'kaiming_init' : [True],
    'optimizer_theta': ["ADAM"],
    'optimizer_mask': ["ADAM"],
    'tensorboard': [True]
}

conditions = [
    ({'x_dep_masking': [True]}, {'optimizer_theta' : ["SGD"],
                                 'optimizer_mask' : ["SGD"],
                                 'val_start':[300],
                                 'test_start':[300],
                                 #'step_size': [0.1],
                                 'x_debug': [True],
                                 'epochs': [600]}),
    ({'x_dep_masking': [False]}, {'x_output_shift' : [0.], 'x_debug': [False],
                                  'x_debug_noise' : [0.]}),
    ({'num_shots_train': [1]}, {'batch_size' : [4], 'mask_lr': [0.01],  'step_size': [0.5]}),
    ({'gradient_steps': [1]}, {'second_order' : [True]}),     
]


"""Define exceptions for the grid search.
Sometimes, not the whole grid should be searched. For instance, if an `SGD`
optimizer has been chosen, then it doesn't make sense to search over multiple
`beta2` values of an Adam optimizer.
Therefore, one can specify special conditions or exceptions.
Note* all conditions that are specified here will be enforced. Thus, **they
overwrite the** :attr:`grid` **options above**.

How to specify a condition? A condition is a key value tuple: whereas as the
key as well as the value is a dictionary in the same format as in the
:attr:`grid` above. If any configurations matches the values specified in the
"key" dict, the values specified in the "values" dict will be searched instead.

Note, if arguments are commented out above but appear in the conditions, the
condition will be ignored.
"""

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
### This code only has to be adapted if you are setting up this template for a
### new simulation script!

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.
import os
_SCRIPT_NAME = os.path.join(os.path.dirname(__file__), 'train.py')

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'

# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword `finished`!.
_SUMMARY_KEYWORDS = [
    # Track all performance measures with respect to the best mean accuracy.
    'mean_sparsity_best',
    'mean_sparsity_end', 
    'best_acc_epoch',
    'best_acc',
    'end_acc',
    'cross_datasets_name',
    'cross_datasets_best_accs',
    'finished'
    ]


# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

def _get_performance_summary(out_dir, cmd_ident):
    """See docstring of method
    :func:`hpsearch.hpsearch._get_performance_summary`.

    You only need to implement this function, if the default parser in module
    :func:`hpsearch.hpsearch` is not sufficient for your purposes.

    In case you would like to use a custom parser, you have to set the
    attribute :attr:`_SUMMARY_PARSER_HANDLER` correctly.
    """
    pass

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = None # Default parser is used.
#_SUMMARY_PARSER_HANDLE = _get_performance_summary # Custom parser is used.

def _performance_criteria(summary_dict, performance_criteria):
    """Evaluate whether a run meets a given performance criteria.

    This function is needed to decide whether the output directory of a run is
    deleted or kept.

    Args:
        summary_dict: The performance summary dictionary as returned by
            :attr:`_SUMMARY_PARSER_HANDLE`.
        performance_criteria (float): The performance criteria. E.g., see
            command-line option `performance_criteria` of script
            :mod:`hpsearch.hpsearch_postprocessing`.

    Returns:
        bool: If :code:`True`, the result folder will be kept as the performance
        criteria is assumed to be met.
    """
    ### Example:
    # return summary_dict['performance_measure1'] > performance_criteria

    raise NotImplementedError('TODO implement')

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE = None
#_PERFORMANCE_EVAL_HANDLE = _performance_criteria

# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = None
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False

# FIXME: This attribute will vanish in future releases.
# This attribute is only required by the `hpsearch_postprocessing` script.
# A function handle to the argument parser function used by the simulation
# script. The function handle should expect the list of command line options
# as only parameter.
# Example:
# >>> from classifier.imagenet import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='cl_ilsvrc_cub',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
_ARGPARSE_HANDLE = None

if __name__ == '__main__':
    pass


