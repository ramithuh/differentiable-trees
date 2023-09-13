import os
import sys

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

import jax.numpy as jnp
import jax.nn as nn

import jax
import optax
import wandb
import argparse
import functools

import pydot
import numpy as np

from typing import Dict
from jaxtyping import Array, Float

from modules.vis_utils import *
from modules.tree_func import *
from modules.gt_tree_gen import *

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

from arg_parser import *
from modules.upgma import differentiable_upgma

## Parse Arguments
args = vars(parse_args())


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
n_letters   = int(args['letters']) if args['letters']!=None else 21

args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule']!=None else [1,0.25,100,100] #start, increment, end, interval
 
 
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
    'lr': args['learning_rate'],
    'epochs': 200000,
    'tLs': args['tree_loss_schedule'] # #tree loss schedule
    }

print(pretty_print_dict(metadata))

wandb.login(key = os.environ.get('WANDB_API_KEY_so'))
wandb.init(project="Differentiable-Trees", name = metadata['exp_name'], entity="ramithx1", config = metadata, tags=["FASRC","UPGMA"], notes = args['notes'])

## JAX Doesn't like some data types when jitting
metadata['exp_name'] = None
metadata['args']['notes'] = None
args['notes'] = None

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

    gt_tree = show_graph_with_labels(tree, n_leaves, True)
    gt_seqs_plot = px.imshow(jnp.argmax(gt_seqs, axis = 2), text_auto=True)

    gt_cost           = compute_cost(gt_seqs, tree, metadata, surrogate_loss = False)
    gt_cost_surrogate = compute_cost(gt_seqs, tree, metadata)
    
    gt_tree_force_loss, gt_loop_loss, gt_bidirection_loss = enforce_graph(tree, 10, metadata)

    wandb.log({
        "epoch": -1, 
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
    't': jax.random.normal(key, (n_leaves,n_leaves)).astype(jnp.bfloat16)
    }

tree_mask = {}

if(args['fix_tree']):
    tree_mask['t'] = 'fixed'
else:
    tree_mask['t'] = 'learn'

## Manually override the parameters such that the updated tree is the same as the base tree
if(args['initialize_tree']):
    params['t'] = tree[0:-1,n_leaves:]*10
    
#### Add the ancestor sequences to the parameters
for i in range(0, n_ancestors):
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

fig2 = show_graph_with_labels(discretize_tree_topology(differentiable_upgma(params['t']), n_all),n_leaves, True)



compute_loss_upgma(params, seqs, base_tree, metadata, metadata['tLs'][0])

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
            t_ = differentiable_upgma(params['t'], [0.1, 1, 0.5 + metadata['tLs'][0]])
            #t_ = update_tree(params, base_tree)
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
    grads = jax.grad(compute_loss_upgma)(params, seqs, base_tree, metadata, metadata['tLs'][0])
    cost, cost_surrogate, tree_force_loss, loop_loss, bidirection_loss, recurse_cost, loss = compute_loss_upgma(params, seqs, base_tree, metadata, metadata['tLs'][0], verbose = True)

    ## Log the metrics
    if(_%1==0):
        wandb.log(
            {"epoch":_, 

            "loss":loss,
            "traversal cost": cost,
            "traversal cost (surrogate)": cost_surrogate,
            "tree force loss": tree_force_loss,
            "loop loss": loop_loss,
            "recusive loss": recurse_cost,
            "bidirection loss": bidirection_loss,
            "tLs":metadata['tLs'][0]})

    if(_%100==0):
        wandb.log({
            "tree":wandb.Image(tree_at_epoch),
            "tree matrix":wandb.data_types.Plotly(tree_matrix_at_epoch),
            "Seq":wandb.data_types.Plotly(new_seq),
            "last ancestor" :  wandb.data_types.Plotly(px.imshow((seqs_[-1]), text_auto=True)),
            })

    if(_%100==0):
        print("epoch ",_, end = " ")
        print(metadata['tLs'][0], end = " ")
        print("{:.5f}".format(loss.item(),end = " \n"))
    
    #=> Update the parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    ### update the tree loss schedule
    if(_%metadata['tLs'][3]==0):
        metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])