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
from helpers import *
from tree_func import *

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

key = jax.random.PRNGKey(42)

#### Define Sequence length and number of leaves
seq_length = 4

n_leaves    = 4
n_ancestors = n_leaves - 1

n_all = n_leaves + n_ancestors

#### Generate a random sequence of 0s and 1s
seqs = jax.random.randint(key, shape = (n_all, seq_length), minval = 0, maxval = 2).astype(jnp.float32)

#### Generate a base tree
base_tree = jnp.zeros((n_all, n_all))


#### Define optimizer and parameters
optimizer = optax.adam(1)
params = {'t': jax.random.normal(key, (n_all - 1,n_ancestors))}

for i in range(0, n_ancestors):
    params[chr(97+i + n_leaves)] = jax.random.normal(key, (seq_length,))
    
opt_state = optimizer.init(params)

print(params.keys())



wandb.init(project="Differentiable-Trees", entity="ramithx1")

fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params, base_tree), n_all),n_leaves, True)

compute_loss(params, seqs, base_tree)

trees = []

for _ in range(1500):
    grads = jax.grad(compute_loss)(params, seqs, base_tree)


    t_= update_tree(params, base_tree)
    t__ = discretize_tree_topology(t_,n_all)
    t__ = t__.at[-1,-1].set(0)

    trees.append(t__)
    
    if(_%100==0):
        # temp  = combine_to_seq(params, seqs)
        loss = compute_loss(params, seqs, base_tree)

        wandb.log({"epoch":_, "loss":loss})

        print("{:.5f}".format(loss,end = " seq = {}\n"))
        #print(t_[:,-2],",",t_[:,-1],end="}\n")
    
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)