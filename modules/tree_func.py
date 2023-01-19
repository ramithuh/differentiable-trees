import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

def discretize_tree_topology(tree, n_nodes):
    t_argmax = jnp.argmax(tree,axis = 1)
    t_q = nn.one_hot(t_argmax, n_nodes)
    
    return t_q

def update_tree(params, base_tree):
    n_leaves = (base_tree.shape[1] - params['t'].shape[1])
    
    t_softmax = nn.softmax(params['t'],axis=1)

    t = base_tree.at[0:-1,n_leaves:].set(t_softmax)
    t = t.at[-1,-1].set(1) ### need to add this (so that the last sequence is matched with itself) 
    
    return t

def combine_to_seq(params, seqs, n_leaves, n_all, temperature = 1):
    
    for i in range(0, n_all - n_leaves):
        key = chr(97+i + n_leaves)
        
        seq      = nn.sigmoid(temperature*params[key])
        hard_seq = (seq > 0.5).astype(int)
        
        seqs = seqs.at[- n_leaves + i + 1].set(seq)
        #seqs = seqs.at[- n_leaves + i + 1].set(jax.lax.stop_gradient(hard_seq - seq) + seq)
    
#     seq_1 = nn.sigmoid(temperature*params['d'])
#     seq_2 = nn.sigmoid(temperature*params['e'])
    
#     hard_seq_1 = (seq_1 > 0.5).astype(int)
#     hard_seq_2 = (seq_2 > 0.5).astype(int)
    
    
    
    
#     seqs = seqs.at[-2].set(seq_1)
#     seqs = seqs.at[-1].set(seq_2)
    
    return seqs

def enforce_graph(t_, s, metadata = None, verbose = False):
    ''' 

        Enforces constraints such that,
            1) the tree is bifurcating
            2) there are no self loops
        
        Args:
            t_: tree topology
            s: scaling factor
            verbose: print the loss values
        Returns:
            loss: loss value
    '''
    n_all = metadata['n_all']
    n_leaves = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']

    tree_force_loss = jnp.sum(jnp.power(s*jnp.abs(jnp.sum(t_[:-1, n_all - n_ancestors: n_all], axis = 0) - 2),2))
    
    loop_loss = jnp.multiply(jnp.identity(n_all-1), t_[:-1,:-1]).sum()*100  ## penalize self connections
    
    bidirection_loss = jnp.sum(jnp.matmul(t_,t_)*jnp.identity(t_.shape[0]))*100 ## penalize bidirectional connections
    
    if(verbose):
        print("bifurcating tree_forcing_loss = ", tree_force_loss)
        print("self loop_loss = ", loop_loss)
        print("bidirectional loss = ", bidirection_loss)
    
    return tree_force_loss, loop_loss, bidirection_loss


def compute_cost(sequences, tree, verbose = False, average = True):
    if(verbose==True):
        show_graph_with_labels(discretize_tree_topology(tree, n_all),n_leaves)
        
    selection = jnp.matmul(tree,sequences)
    # print(selection)
    
#     #method 1: (downside needs integers as we are indexing)
#     selection = selection.astype(int)
#     ans = sm[selection,sequences].sum()
    
    #method 2:
    if(average):
        ans = abs(selection - sequences).sum() ## works currently as sm is the basic config. need to think about extending
    else:
        ans = jnp.sum(abs(selection - sequences), axis = 1)
    
    return ans

def compute_loss(params, seqs, base_tree, metadata, verbose = False):
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']
    
    seqs_ = combine_to_seq(params, seqs, n_leaves, n_all)
    t_    = update_tree(params, base_tree)

    
    cost = compute_cost(seqs_,t_)
    tree_force_loss, loop_loss, bidirection_loss = enforce_graph(t_,10,metadata)
    
    loss = cost + tree_force_loss + loop_loss + bidirection_loss

    if(verbose):
        return cost, tree_force_loss, loop_loss, bidirection_loss, loss

    return loss