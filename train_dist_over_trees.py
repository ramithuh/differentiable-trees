from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

import os
import sys

import optax
import jaxopt

import wandb
import pydot
import warnings
import argparse
import functools
from typing import Dict

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

import jax
## use cpu for the time being, let's see whether user requested a gpu later.
jax.config.update("jax_default_device", jax.devices("cpu")[0]) 
import jax.nn as nn
from jax import vmap
import jax.numpy as jnp
from jax.lib import xla_bridge
from jaxtyping import Array, Float

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

from modules.vis_utils import *
from modules.tree_func import *
from modules.gt_tree_gen import *
from modules.sankoff import *
from arg_parser_v2 import *


def generate_vmap_keys(params):
    vmap_keys = {}
    
    for key in params.keys():
        if(key!='t'):
            vmap_keys[key] = None
        else:
            vmap_keys[key] = 0
        
    return vmap_keys

@jit
def get_one_tree_and_seq(params, pos):
    print(params.keys())
    new_params = {}
    
    ## extract the specified tree
    new_params['t'] = params['t'][pos]
    
    for i in range(0, len(params.keys()) - 1):
        # there's only one set of sequences, so no need to index
        new_params[str(i)] = params[str(i)]
    
    return new_params

## Parse Command Line Arguments and perform checks
args = vars(parse_args_v2())
args = sanity_check(args)

#### Define Sequence length and number of leaves
seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20
n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors
n_mutations = int(args['mutations']) if args['mutations']!=None else 3
n_letters   = int(args['letters']) if args['letters']!=None else 20

args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule']!=None else [0,0.01,100,5]
 
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
    'lr_seq' : args['learning_rate_seq'] if args['learning_rate_seq']!=None else args['learning_rate']*10,
    'epochs': args['epochs'],
    'tLs': args['tree_loss_schedule'],
    'init_count' : args['init_count'] if args['init_count']!=None else 1,
    'initializer' : args['initializer']
    }

wandb.login(key = os.environ.get('WANDB_API_KEY_so'))
wandb.init(project=args['project'], name = args['notes'] + metadata['exp_name'], entity="ramithx1", config = metadata, tags=["FASRC","Fixed Groundtruth", "Fixed GT Seqs Order", args['tags']], notes = args['notes'])

if(args['gpu']!=None):
    print_critical_info(f"Utilizing gpu -> {args['gpu']} \n")
    jax.config.update("jax_default_device", jax.devices("gpu")[args['gpu']])
else:
    if(xla_bridge.get_backend().platform == "gpu"):
        print_critical_info("There's a gpu available, but you didn't specify to use it 😏. So using cpu instead 🤷🏻‍♂️. \n")    
        print(f"Available GPUs: {jax.devices('gpu')}")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

print(pretty_print_dict(metadata))

## JAX Doesn't like some data types when jitting
metadata['exp_name'] = None
metadata['args']['notes'] = None
metadata['notes'] = None
metadata['tags']  = None
metadata['project'] = None
init_func = metadata['initializer'][:]
metadata['initializer'] = None
args['notes'] = None
args['tags'] = None
args['project'] = None
args['initializer'] = None

#### Generate a random sequence of 0s and 1s
key = jax.random.PRNGKey(metadata['seed'])
offset = 10

#### Generate a base tree
base_tree = jnp.zeros((n_all, n_all))

sm = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.float64)

if(args['groundtruth']):
    # seqs, gt_seqs, tree = generate_groundtruth(metadata, metadata['seed'])
    
    [seqs, leaf_ids], [gt_seqs, label_names], [tree, shrinked_tree, original_tree, fig] = generate_general_groundtruth(metadata, perfect=False, seed = metadata['seed'], show_derivation = True)

    print("_____")
    print(seqs)
    print("gt seqs")
    print(gt_seqs)
    print("____")

    seqs    = jax.nn.one_hot(seqs, n_letters).astype(jnp.float64)
    gt_seqs = jax.nn.one_hot(gt_seqs, n_letters).astype(jnp.float64)

    #if we don't want sequences to change, set seqs to gt_seqs
    if(args['fix_seqs'] or args['initialize_seq']):
        seqs = gt_seqs

    #if we don't want tree to change, set base_tree to tree
    if(args['fix_tree']):
        base_tree = tree
        
    if(not(args['fix_seqs'])):
        ## since we know the gt tree, let's get the real ancestors using sankoff algorithm! 🎉 
        ## This will help us to see whether there's a better groundtruth ancestors for this tree
        cost_mat = (jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters)).astype(jnp.float64)
        
        print("running sankoff on groundtruth tree")
        sankoff_seqs, dp, sankoff_cost = run_sankoff(tree, cost_mat, jnp.argmax(seqs, axis = 2), metadata, return_path = True)
        print("done running sankoff on groundtruth tree")
        
        wandb.log({'sankoff_cost': sankoff_cost})
        
        if(metadata['n_leaves'] <= 64 ):
            wandb.log({
                'sankoff_seqs': wandb.data_types.Plotly(px.imshow(sankoff_seqs, text_auto=True)),
                # 'dp_table' : wandb.data_types.Plotly(px.imshow(dp[:,:,0], text_auto=True)),
                'cost_for_sankoff_seqs' : compute_cost(jax.nn.one_hot(sankoff_seqs, n_letters).astype(jnp.float64), tree, sm)
            })
        
    if(args['shuffle_seqs']):        
        shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False) 
        seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
        
        shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False) 
        seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)

    gt_tree = show_graph_with_labels(tree, n_leaves, return_img = True)
    gt_seqs_plot = px.imshow(jnp.argmax(gt_seqs, axis = 2), text_auto=True)

    gt_tree_matrix = px.imshow(tree, text_auto=True)
    
    gt_cost           = compute_cost(gt_seqs, tree, sm)
    gt_cost_surrogate = compute_surrogate_cost(gt_seqs, tree)
    
    gt_tree_force_loss = enforce_graph(tree, 10, metadata)

    wandb.log({
        "Groundtruth Tree" : wandb.Image(gt_tree),
        "Groundtruth Tree Matrix" : wandb.data_types.Plotly(gt_tree_matrix),
        "Groundtruth Seq" : wandb.data_types.Plotly(gt_seqs_plot),

        "Groundtruth Tree Force Loss" : gt_tree_force_loss,
        "Groundtruth Traversal Cost (surrogate)" : gt_cost_surrogate,
        "Groundtruth Traversal Cost" : gt_cost,
        "Groundtruth Total Loss" : gt_tree_force_loss + gt_cost
        })


#### Define parameters (t and the ancestor sequences)

function_call = f"jax.nn.initializers.{init_func}"
initializer = eval(function_call)

# initializer = jax.nn.initializers.orthogonal(10)
# initializer = jax.nn.initializers.normal(1)

#jax.nn.initializers.kaiming_normal()
#jax.nn.initializers.orthogonal()
#jax.nn.initializers.kaiming_normal()
#jax.nn.initializers.orthogonal()


## Tree distribution : count = metadata['init_count']
params : Dict[str, Array] = {
    't': initializer(key + offset, (metadata['init_count'], n_all - 1,n_ancestors), jnp.float64)
    }


## Manually override the parameters such that the updated tree is the same as the base tree
if(args['initialize_tree']):
    print_critical_info("Initializing tree using groundtruth tree \n")
    
    params['t'] = jnp.zeros((metadata['init_count'], n_all - 1,n_ancestors))
    params['t'] = params['t'].at[:,].set(tree[0:-1,n_leaves:]*100)

print("HMMM")
print(params['t'][0])
print("end HMMM")
    
#### Add the ancestor sequences to the parameters (only one set for all trees)
for i in range(0, n_ancestors):
    
    if(args['initialize_seq']):
        if(n_leaves > 1024):
            raise NotImplementedError("Sankoff backtracking not implemented for n_leaves > 1024 due to execution time")
        else:
            print_critical_info("Initializing sequences using sankoff ancestors \n")
            params[str(i)] = jax.nn.one_hot(sankoff_seqs[n_leaves + i], n_letters).astype(jnp.float64)*100
    else:
        # just one set of sequences
        params[str(i)] = initializer(key+i+offset, (seq_length, n_letters), jnp.float64)
        #jax.random.normal(key+i+offset, (seq_length, n_letters)).astype(jnp.float64)
    
#### Initialize the optimizer

#eps_root = 1e-8, eps = 1e-8

optimizer = optax.adam(metadata['lr'])
opt_state = optimizer.init(params)

## Organize how inputs are passed to the usual `compute_detailed_loss_optimized` function (i.e. parallelize over the multiple trees)
# Generate dictionary for params; tree -> parallelize, seqs -> don't
vmap_keys = generate_vmap_keys(params)
mapped_compute_loss_optimized = jax.vmap(compute_loss_optimized, in_axes = (vmap_keys, None, None, None, None), out_axes = 0)

def tree_distribution_loss(params, seqs, metadata, temp, epoch):
    val = mapped_compute_loss_optimized(params, seqs, metadata, temp, epoch)
    loss = jnp.mean(val)
    return loss 

vmap_compute_detailed_loss_optimized =  jit(vmap(compute_detailed_loss_optimized, (vmap_keys, None, None, None, None, None),0))
vmap_update_tree = vmap(update_tree, (vmap_keys, None, None), 0)
vmap_discretize_tree_topology = vmap(discretize_tree_topology, (0, None), 0)

fixed_dummy_pos = 0 

## Get one set of tree and seq
params_ = get_one_tree_and_seq(params, fixed_dummy_pos)
fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params_), n_all),n_leaves, return_img = True)
compute_loss(params_, seqs, base_tree, metadata, metadata['tLs'][0])


best_ans = 1e9
best_seq = None
best_tree = None

pos = 0


@jax.jit
def _norm_msa_grad(g):
  # g.shape = (N, L, A)
  gn = jnp.linalg.norm(g,axis=(-1,-2),keepdims=True) # shape = (N,1,1)
  g = g * jnp.sqrt(g.shape[1]) / (gn + 1e-7) 
  return g

@jax.jit
def scale_grads(grad):
    # scale gradients
    new_grads = grad.copy()
    new_grads['t'] = grad['t'] #copy gradients of trees unchanged
    
    len_ = len(grad.keys()) - 1
    
    # first make a tensor out of the individual elements.
    shape = grad['0'].shape 
    

    # msa = (N, L, A)
    msa = jnp.empty((len_, shape[0], shape[1]))
    for i in range(len_):
        temp = grad[str(i)]
        msa = msa.at[i].set(temp)
        
    msa_norm_grad = _norm_msa_grad(msa)
    
    for i in range(len_):
        #copying corresponding normalized gradients to the new dictionary
        temp = msa_norm_grad[i]
        new_grads[str(i)] = temp
    
    return new_grads

#### Optimization loop
for _ in range(metadata['epochs']):

    if(_%50==0):
        ###~ Get the current discretized tree
        if(args['fix_tree']): 
            t_ = base_tree
        else:
            t_ = vmap_update_tree(params, _, metadata['tLs'][0])
        t_d           = vmap_discretize_tree_topology(t_,n_all)
        tree_at_epoch        = show_graph_with_labels(t_d[0], n_leaves, return_img = True)
        tree_matrix_at_epoch = [wandb.Image(np.asarray(t_[idx])) for idx in range(0, t_.shape[0])]

        ###~ Get the current sequences as a plot
        if(args['fix_seqs']):
            seqs_ = seqs
        else:
            seqs_ = update_seq(params, seqs, metadata['seq_temp'])
        new_seq = px.imshow(jnp.argmax(seqs_, axis = 2), text_auto=True)
        
    
    #compute gradients
    grads = jax.grad(tree_distribution_loss)(params, seqs, metadata, metadata['tLs'][0], _)
    if(args['normalize_gradients']):
        grads = scale_grads(grads)
   
    cost, cost_surrogate, tree_force_loss, entropy_loss, loss = vmap_compute_detailed_loss_optimized(params, seqs, metadata, metadata['tLs'][0], sm,  _)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    params             = optax.apply_updates(params, updates)
    
    # extracting best tree for logging
    pos = jnp.argmin(cost)
    params_ = get_one_tree_and_seq(params,pos)
    
    if(cost.min() < best_ans):
        if(_%100==0):
            print_success_info("Found a better tree at epoch %d with cost %f from tree %d. (delta at epoch = %d) \n" % (_, cost.min(), pos, cost.max()-cost.min()))
        best_ans = cost.min()
        
        t_ = update_tree(params_, _, metadata['tLs'][0])
        best_tree = discretize_tree_topology(t_,n_all)
        best_seq = update_seq(params_, seqs, metadata['seq_temp'])
        
        ## Log the metrics
    if(_%50==0):
        wandb.log(
            {"epoch":_, 

            "loss":loss[pos],
            "traversal cost": cost[pos],
            "traversal cost (surrogate)": cost_surrogate[pos],
            "tree force loss": tree_force_loss[pos],
            "entropy loss": entropy_loss[pos],
            
            "tree":wandb.Image(tree_at_epoch),
            "tree matrix":[curr_tree for curr_tree in tree_matrix_at_epoch] ,
            "Seq":wandb.data_types.Plotly(new_seq),
            "last ancestor" :  wandb.data_types.Plotly(px.imshow((seqs_[-1]), text_auto=True)),
            "tLs":metadata['tLs'][0],})

    if(_%50==0):
        print_bold_info(f"epoch {_}")
        print("{:.3f}".format(metadata['tLs'][0]), end = " ")
        print("{:.3f}".format(loss[pos].item()), end=" ")
        print("{:.3f}".format(cost[pos].item()), end = "\n")
    

    ### update the tree loss schedule
    if(_%metadata['tLs'][3]==0 and _ > args['delay_schedule']):
        metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])
        
print_success_info("Optimization done!\n")
print_success_info("Final cost: {:.5f}\n".format(cost[pos]))
print_success_info("Best cost encountered: {:.5f}\n".format(best_ans))

if(args['fix_tree']):
    print_success_info("Sankoff cost for groundtruth tree: {:.5f}\n".format(sankoff_cost.item()))
    target_cost = sankoff_cost
elif(args['fix_seqs']):
    print_success_info("Groundtruth tree cost: {:.5f}\n".format(gt_cost))
    target_cost = gt_cost
else:
    print_critical_info("No groundtruth to compare to\n")
    target_cost = 0


if(abs(target_cost - best_ans) == 0):
    print_success_info("Optimization succeeded! Reached groundtruth 🚀\n")
    wandb.log({"success":True, "Error" : 0})
else:
    wandb.log({"success":False, "Error" : (best_ans - sankoff_cost)})

best_tree_img = show_graph_with_labels(best_tree, n_leaves, return_img = True)
best_tree_adj = px.imshow(best_tree, text_auto=True)

wandb.log({
        "best cost":best_ans, 
        "best_tree_adj" : wandb.data_types.Plotly(best_tree_adj),
        "best_tree" : wandb.Image(best_tree_img),
        "best_seq" : wandb.data_types.Plotly(px.imshow(jnp.argmax(best_seq, axis = 2), text_auto=True))
           })

wandb.finish()