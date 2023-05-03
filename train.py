import os
import sys

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

import jax.numpy as jnp
import jax.nn as nn

import jax
jax.config.update("jax_default_device", jax.devices("cpu")[0]) ## use cpu for the time being 

import optax
import wandb
import argparse
import functools
import warnings

import pydot
import numpy as np

from typing import Dict
from jaxtyping import Array, Float

from modules.vis_utils import *
from modules.tree_func import *
from modules.gt_tree_gen import *
from modules.sankoff import *

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

from arg_parser import *

def invert_dict(dictionary):
  new_dict = dictionary
  for key in dictionary.keys():
    if(new_dict[key] == 'fixed'):
      new_dict[key] = 'learn'
    else:
      new_dict[key] = 'fixed'

  return new_dict

#### Parse arguments
parser = argparse.ArgumentParser(description='Train a differentiable tree model. \n sample usage :  python train.py -l 8 -m 10 -sl 50 -fs')

parser.add_argument('-gt', '--groundtruth', action='store_true', help='Retrieve groundtruth?')
parser.set_defaults(groundtruth=True)


parser.add_argument('-n','--notes'       , help='notes'            , required=False, type = str, default = '')
parser.add_argument('-t','--tags'        , help='notes'            , required=False, type = str, default = 'default')

#tree and seq generate parameters
parser.add_argument('-l','--leaves'      , help='# of leaves?'    , required=False, type = int)
parser.add_argument('-m','--mutations'   , help='# of leaves?'    , required=False, type = int)
parser.add_argument('-sl','--seq_length' , help='length of seq'   , required=False, type = int)
parser.add_argument('-nl','--letters'    , help='# of letters?'    , required=False, type = int)
parser.add_argument('-s','--seed'        , help='seed'            , required=False, type = int)

parser.add_argument('-ai','--alternate_interval', help='alternate_interval', required=False, type = int)
parser.add_argument('-tLs','--tree_loss_schedule' , help='tree loss schedule', required=False, type = str)
parser.add_argument('-lr','--learning_rate'      , help='learning rate'    , required=False, type = float)

## what to train?
parser.add_argument('-fs', '--fix_seqs', action='store_true', help='Fix sequences when training ?')
parser.set_defaults(fix_seqs=False)

parser.add_argument('-shs', '--shuffle_seqs', action='store_true', help='Shuffle groundtruth sequences?')
parser.set_defaults(shuffle_seqs=False)

parser.add_argument('-ft', '--fix_tree', action='store_true', help='Fix tree when training ?')
parser.set_defaults(fix_tree=False)

parser.add_argument('-it', '--initialize_tree', action='store_true', help='Initialize Tree with groundtruth?')
parser.set_defaults(initialize_tree=False)

parser.add_argument('-is', '--initialize_seq', action='store_true', help='Initialize Seq with groundtruth?')
parser.set_defaults(initialize_seq=False)

parser.add_argument('-alt', '--alternate_optimization', action='store_true', help='Alternate tree and seq optimization?')
parser.set_defaults(alternate_optimization=False)

parser.add_argument('-g','--gpu', help='specify device', required=False, type = int)

args = vars(parser.parse_args())

if(args['gpu']!=None):
    print(f"using gpu -> {args['gpu']}")
    jax.config.update("jax_default_device", jax.devices("gpu")[args['gpu']])
else:
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

### set contradictory arguments (otherwise could be confusing in wandb)
if(args['fix_seqs']):
    args['initialize_seq'] = True #if we fix seqs, automatically this is true

if(args['fix_tree']):
    args['initialize_tree'] = True  #if we fix tree, automatically this is true

#### Define Sequence length and number of leaves
seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20

n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors
n_mutations = int(args['mutations']) if args['mutations']!=None else 3
n_letters   = int(args['letters']) if args['letters']!=None else 20

args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule']!=None else [0,0.02,1,50]
 
metadata = {
    'n_all' : n_all,
    'n_leaves' : n_leaves, 
    'n_ancestors' : n_ancestors,
    'seq_length' : seq_length,
    'n_letters' : n_letters,
    'n_mutations' : n_mutations,
    'args': args,
    'exp_name' : f"l={n_leaves}, m={n_mutations}, s={seq_length}, fs={args['fix_seqs']}, ft={args['fix_tree']}" ,
    'seed' : int(args['seed']) if args['seed']!=None else 42,
    'seq_temp': 0.5,
    'lr': 0.001 if args['learning_rate']==None else args['learning_rate'],
    'epochs': 10000,
    'tLs': args['tree_loss_schedule'] # #tree loss schedule
    }

wandb.login(key = os.environ.get('WANDB_API_KEY_so'))
wandb.init(project="Results-summary", name = args['notes'] + metadata['exp_name'], entity="ramithx1", config = metadata, tags=["FASRC","Fixed Groundtruth", "Fixed GT Seqs Order", args['tags']], notes = args['notes'])

print(pretty_print_dict(metadata))

## JAX Doesn't like some data types when jitting
metadata['exp_name'] = None
metadata['args']['notes'] = None
metadata['notes'] = None
metadata['tags'] = None
args['notes'] = None
args['tags'] = None

#### Generate a random sequence of 0s and 1s
key = jax.random.PRNGKey(metadata['seed'])
seqs = jax.random.randint(key, shape = (n_all, seq_length, n_letters), minval = 0, maxval = 2).astype(jnp.bfloat16)

#### Generate a base tree
base_tree = jnp.zeros((n_all, n_all))


if(args['groundtruth']):
    seqs, gt_seqs, tree = generate_groundtruth(metadata, metadata['seed'])

    seqs    = jax.nn.one_hot(seqs, n_letters).astype(jnp.bfloat16)
    gt_seqs = jax.nn.one_hot(gt_seqs, n_letters).astype(jnp.bfloat16)

    #if we don't want sequences to change, set seqs to gt_seqs
    if(args['fix_seqs'] or args['initialize_seq']):
        seqs = gt_seqs

    #if we don't want tree to change, set base_tree to tree
    if(args['fix_tree']):
        base_tree = tree
        
    if(True):
        ## since we know the gt tree, let's get the real ancestors using sankoff algorithm! 🎉 
        ## This will help us to see whether there's a better groundtruth ancestors for this tree
        cost_mat = jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters)
        
        print("running sankoff on groundtruth tree")
        sankoff_seqs, _ = run_sankoff(tree, cost_mat, jnp.argmax(seqs, axis = 2), metadata)
        print("done running sankoff on groundtruth tree")
        
        wandb.log(
            {
                'sankoff_seqs': wandb.data_types.Plotly(px.imshow(sankoff_seqs, text_auto=True)),
                'sankoff_cost': _,
                'cost_for_sankoff_seqs' : compute_cost(jax.nn.one_hot(sankoff_seqs, n_letters).astype(jnp.bfloat16), tree, metadata, surrogate_loss = False)
            })
        
    if(args['shuffle_seqs']):        
        shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False) 
        seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
        
        shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False) 
        seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)

    gt_tree = show_graph_with_labels(tree, n_leaves, True)
    gt_seqs_plot = px.imshow(jnp.argmax(gt_seqs, axis = 2), text_auto=True)

    gt_cost           = compute_cost(gt_seqs, tree, metadata, surrogate_loss = False)
    gt_cost_surrogate = compute_cost(gt_seqs, tree, metadata)
    
    gt_tree_force_loss, gt_loop_loss, gt_bidirection_loss = enforce_graph(tree, 10, metadata)

    wandb.log({
        "Groundtruth Tree" : wandb.Image(gt_tree),
        "Groundtruth Seq" : wandb.data_types.Plotly(gt_seqs_plot),

        "Groundtruth Tree Force Loss" : gt_tree_force_loss,
        "Groundtruth Loop Loss" : gt_loop_loss,
        "Groundtruth Bidirection Loss" : gt_bidirection_loss,
        "Groundtruth Traversal Cost (surrogate)" : gt_cost_surrogate,
        "Groundtruth Traversal Cost" : gt_cost,
        "Groundtruth Total Loss" : gt_tree_force_loss + gt_loop_loss + gt_bidirection_loss + gt_cost
        })


#### Define parameters (t and the ancestor sequences)
params : Dict[str, Array] = {
    't': jax.random.normal(key, (n_all - 1,n_ancestors))
    }


tree_mask = {}

if(args['fix_tree']):
    tree_mask['t'] = 'fixed'
else:
    tree_mask['t'] = 'learn'

## Manually override the parameters such that the updated tree is the same as the base tree
if(args['initialize_tree']):
    warnings.warn("Initializing tree using groundtruth tree", UserWarning)
    params['t'] = tree[0:-1,n_leaves:]*10
    
#### Add the ancestor sequences to the parameters
for i in range(0, n_ancestors):
    
    if(args['initialize_seq']):
        if(n_leaves > 1024):
            raise NotImplementedError("Sankoff backtracking not implemented for n_leaves > 1024 due to execution time")
        else:
            warnings.warn("Initializing sequences using sankoff ancestors", UserWarning)
            params[str(i)] = jax.nn.one_hot(sankoff_seqs[n_leaves + i], n_letters).astype(jnp.bfloat16)*100
    else:
        params[str(i)] = jax.random.normal(key+i, (seq_length, n_letters)).astype(jnp.bfloat16)

    if(args['alternate_optimization'] or args['fix_seqs']):
        tree_mask[str(i)]  = 'fixed'
    else:
        tree_mask[str(i)]  = 'learn'
    
#### Initialize the optimizer
# optimizer = optax.adam(metadata['lr'])
optimizer = optax.multi_transform(
                    {
                     'learn': optax.adam(metadata['lr']), 
                     'fixed': optax.set_to_zero()
                     },
                    tree_mask
                    )

print("parameters to optimize:", end = " -> ")
print(params.keys())
print(tree_mask.keys())

opt_state = optimizer.init(params)

fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params, base_tree), n_all),n_leaves, True)



compute_loss(params, seqs, base_tree, metadata, metadata['tLs'][0])


#### Training loop
for _ in range(metadata['epochs']):

    if(args['alternate_optimization']):
        if(_ % args['alternate_interval'] == 0 and _>0):
            #if we are in ALT mode switch the mask
            tree_mask = invert_dict(tree_mask)
            print(" mask: ", tree_mask)

            optimizer = optax.multi_transform(
                        {
                        'learn': optax.adam(metadata['lr']), 
                        'fixed': optax.set_to_zero()
                        },
                        tree_mask
                        )
            
            opt_state = optimizer.init(params)

    if(_%100==0):
        ###~ Get the current discretized tree
        if(args['fix_tree']): 
            t_ = base_tree
        else:
            t_ = update_tree(params, base_tree)
        t_d           = discretize_tree_topology(t_,n_all)
        tree_at_epoch        = show_graph_with_labels(t_d, n_leaves, True)
        tree_matrix_at_epoch = px.imshow(t_, text_auto=True)

        ###~ Get the current sequences as a plot
        if(args['fix_seqs']):
            seqs_ = seqs
        else:
            seqs_ = update_seq(params, seqs, metadata['seq_temp'])
        new_seq = px.imshow(jnp.argmax(seqs_, axis = 2), text_auto=True)

    ## Compute the loss and the metrics for loggings purposes
    ##=> sort of forward model (using params to generate new sequences and tree)
    #=> Compute the gradients
    grads = jax.grad(compute_loss)(params, seqs, base_tree, metadata, metadata['tLs'][0])
    cost, cost_surrogate, tree_force_loss, loop_loss, bidirection_loss, recurse_cost, loss = compute_loss(params, seqs, base_tree, metadata, metadata['tLs'][0], verbose = True)

    ## Log the metrics
    if(_%100==0):
        wandb.log(
            {"epoch":_, 

            "loss":loss,
            "traversal cost": cost,
            "traversal cost (surrogate)": cost_surrogate,
            "tree force loss": tree_force_loss,
            "loop loss": loop_loss,
            "recusive loss": recurse_cost,
            "bidirection loss": bidirection_loss,

            "tree":wandb.Image(tree_at_epoch),
            "tree matrix":wandb.data_types.Plotly(tree_matrix_at_epoch),
            "Seq":wandb.data_types.Plotly(new_seq),
            "last ancestor" :  wandb.data_types.Plotly(px.imshow((seqs_[-1]), text_auto=True)),
            "tLs":metadata['tLs'][0],})

    if(_%100==0):
        print("epoch ",_, end = " ")
        print(metadata['tLs'][0], end = " ")
        print("{:.5f}".format(loss,end = " \n"))
    
    #=> Update the parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    ### update the tree loss schedule
    if(_%metadata['tLs'][3]==0):
        metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])