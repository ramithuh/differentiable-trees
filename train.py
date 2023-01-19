import os
import sys

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

import jax.numpy as jnp
import jax.nn as nn
import jax
import optax
import wandb
import functools

import pydot
import numpy as np
from modules.vis_utils import *
from modules.tree_func import *

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

key = jax.random.PRNGKey(42)

#### Define Sequence length and number of leaves
seq_length  = 4

n_leaves    = 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors

metadata = {
    'n_all' : n_all,
    'n_leaves' : n_leaves, 
    'n_ancestors' : n_ancestors 
    }

#### Generate a random sequence of 0s and 1s
seqs = jax.random.randint(key, shape = (n_all, seq_length), minval = 0, maxval = 2).astype(jnp.float32)

#### Generate a base tree
base_tree = jnp.zeros((n_all, n_all))

#### Define optimizer and parameters (t and the ancestor sequences)
optimizer = optax.adam(1)
params = {
    't': jax.random.normal(key, (n_all - 1,n_ancestors))
    }

#### Add the ancestor sequences to the parameters
for i in range(0, n_ancestors):
    params[chr(97+i + n_leaves)] = jax.random.normal(key, (seq_length,))

#### Initialize the optimizer
opt_state = optimizer.init(params)


print("parameters to optimize:", end = " -> ")
print(params.keys())


wandb.init(project="Differentiable-Trees", entity="ramithx1", config = metadata, tags=["FASRC"])

fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params, base_tree), n_all),n_leaves, True)
compute_loss(params, seqs, base_tree, metadata)


#### Training loop
for _ in range(300):
    grads = jax.grad(compute_loss)(params, seqs, base_tree, metadata)

    ## Get the current tree
    t_  = update_tree(params, base_tree)
    t__ = discretize_tree_topology(t_,n_all)
    tree_at_epoch = show_graph_with_labels(t__, n_leaves, True)
    t__ = t__.at[-1,-1].set(0)

    ## Get the current sequences
    new_seq = combine_to_seq(params, seqs, n_leaves, n_all)
    new_seq = px.imshow(new_seq, text_auto=True)
    
    cost, tree_force_loss, loop_loss, bidirection_loss, loss = compute_loss(params, seqs, base_tree, metadata, verbose = True)

    ## Log the metrics
    wandb.log(
        {"epoch":_, 

         "loss":loss,
         "traversal cost": cost,
         "tree force loss": tree_force_loss,
         "loop loss": loop_loss,
         "bidirection loss": bidirection_loss,

         "tree":wandb.Image(tree_at_epoch),
         "Seq":wandb.data_types.Plotly(new_seq)})

    if(_%100==0):
        print("{:.5f}".format(loss,end = " \n"))
    
    #### Update the parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)