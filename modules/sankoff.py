import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp

from typing import Dict, List
from jaxtyping import Array, Float
from functools import partial

# @partial(jax.jit, static_argnums=[5,6]) 
def run_dp(adj : Float[Array, "nodes nodes"], dp, dp_nodes, seq, cost_mat, n_letters, verbose = False):
    '''
        Run sankoff algorithm given the tree topology and the leaf sequences
        
        Args:
            adj: adjacency matrix of the tree
            dp : blank dp table
            dp_nodes : blank dp table for storing the nodes
            seq : leaf sequences
            args : metadata
            verbose : print the progress of the algorit hm
            
        Returns:
            dp : filled dp table
            dp_nodes : node information for backtracking
    '''
    
    n_all    = adj.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_leaves): # initailize the dp table
        dp =  dp.at[i,seq[i].astype(int)].set(0)


    for node in range(n_leaves, n_all): ## loop all the ancestors in order
        ind = jnp.argpartition(adj[:,node], -1)[-2:] ## find the children of the current node

        left_node  = ind[0]
        right_node = ind[1]

        if(verbose):
            print(f"at node {node+1} children are : {left_node+1} and {right_node+1}")

        #for i in range(n_letters): # consider all the possible letters
            
        left_cost_array = cost_mat[::][::] + dp[left_node][::]
        left_cost = jnp.min(left_cost_array, axis = 1)
        left_char = jnp.argmin(left_cost_array, axis = 1)

        right_cost_array = cost_mat[::][::] + dp[right_node][::]
        right_cost = jnp.min(right_cost_array, axis = 1)
        right_char = jnp.argmin(right_cost_array, axis = 1)

        dp = dp.at[node,::].set(left_cost + right_cost)

        dp_nodes = dp_nodes.at[node, ::, 0].set(left_node)
        dp_nodes = dp_nodes.at[node, ::, 1].set(left_char)

        dp_nodes = dp_nodes.at[node, ::, 2].set(right_node)
        dp_nodes = dp_nodes.at[node, ::, 3].set(right_char)

            # dp_nodes[...,node][i] = {str(left_node) : left_char, str(right_node) : right_char}

    return dp, dp_nodes

def run_diff_dp(adj : Float[Array, "nodes nodes"], dp, seq, cost_mat, n_letters, epsilon = 1e-5, verbose = False):
    '''
        Run sankoff algorithm given the tree topology and the leaf sequences
        
        Args:
            adj: adjacency matrix of the tree
            dp : blank dp table
            seq : leaf sequences
            args : metadata
            verbose : print the progress of the algorit hm
            
        Returns:
            dp : filled dp table
            dp_nodes : <TODO>
    '''
    
    n_all    = adj.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_leaves): # initailize the dp table
        dp =  dp.at[i,seq[i].astype(int)].set(0) ## sequences are fixed!! so no need to propogate the gradients till then


    for node in range(n_leaves, n_all): ## loop all the ancestors in order
        ind = jnp.argpartition(adj[:,node], -1)[-2:] ## find the children of the current node

        left_node  = ind[0]
        right_node = ind[1]

        if(verbose):
            print(f"at node {node+1} children are : {left_node+1} and {right_node+1}")

        #for i in range(n_letters): # consider all the possible letters
            
        left_cost_array = cost_mat[::][::] + dp[left_node][::]
        left_cost = -nn.logsumexp(-left_cost_array/epsilon, axis = 1)*epsilon #jnp.min(left_cost_array, axis = 1) #

        right_cost_array = cost_mat[::][::] + dp[right_node][::]
        right_cost = -nn.logsumexp(-right_cost_array/epsilon, axis = 1)*epsilon #jnp.min(right_cost_array, axis = 1) #

        dp = dp.at[node,::].set(left_cost + right_cost)

    return dp

vectorized_dp      = jax.vmap(run_dp, (None, 0, 0, 1, None, None, None), 0)
vectorized_diff_dp = jax.vmap(run_diff_dp, (None, 0, 1, None, None, None, None), 0)


def backtrack_dp(node, letter, seq_chars, connections, n_leaves):
    '''
        Derive the ancestor sequences from the dp table and the node information
        
        Args:
            node : current node
            letter : letter used for current node
            seq_chars : blank sequence table (leaf sequences are already filled)
            connections : node information
            args : metadata
        
        Return : 
            seq_chars : filled sequence table (computed ancestor sequences)
    '''

    if(node < n_leaves): #break if it's a leaf
        return seq_chars


    seq_chars = seq_chars.at[node].set(letter)

    child = connections[node,letter]

    seq_chars = backtrack_dp(child[0].astype(int), child[1].astype(int), seq_chars, connections, n_leaves)
    seq_chars = backtrack_dp(child[2].astype(int), child[3].astype(int), seq_chars, connections, n_leaves)

    return seq_chars

def run_diff_sankoff(adj, cost_mat, seq, metadata, epsilon = 1e-5):
    adj = adj.at[-1,-1].set(0) ##manually remove self connection (convention used in the train script)
    
    n_letters = metadata['n_letters']
    n_leaves  = metadata['n_leaves']
    
    ##empty dp table
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], n_letters))*1e7
    
    # connections = dp_nodes.copy()
        
    dp = vectorized_diff_dp(adj, dp, seq, cost_mat, n_letters, epsilon, False)

    return dp, dp[:, -1].min(axis = 1).sum()

def run_sankoff(adj, cost_mat, seq, metadata, return_path = True):
    adj = adj.at[-1,-1].set(0) ##manually remove self connection (convention used in the train script)
    
    n_letters = metadata['n_letters']
    n_leaves  = metadata['n_leaves']
    
    ##empty dp tables
    dp_nodes = jnp.zeros((seq.shape[1], metadata['n_all'], n_letters, 4))
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], n_letters))*1e7
    
    # connections = dp_nodes.copy()
        
    dp, connections = vectorized_dp(adj, dp, dp_nodes, seq, cost_mat, n_letters, False)
    # print("ran dp for all positions through vmap")


    # print(dp)
    # for i in range(0,seq.shape[1]):
    #     print("ran dp for position", i)
    #     ans = run_dp(adj, dp[i], dp_nodes[i], seq[:,i], cost_mat, n_letters, False)
        
    #     dp       = dp.at[i].set(ans[0])
    #     connections = connections.at[i].set(ans[1])
        
    
    seq_chars = jnp.zeros((seq.shape[1], metadata['n_all'],1)) 
    found_seq = seq.copy().astype(jnp.float32)
    
    if(return_path):
        node = adj.shape[0] - 1

        if(seq.shape[0] <=  1024): # if too many sequences don't backtrack and find best ancestors, cost is enough
            for i in range(0,seq.shape[1]):
                letter = jnp.argmin(dp[i,node,:]).astype(int)

                chars = backtrack_dp(node, letter, seq_chars[i], connections[i], n_leaves).reshape(metadata['n_all'],)
                
                found_seq = found_seq.at[metadata['n_leaves']:,i].set(chars[metadata['n_leaves']:])
                # print(".", end = "")

    
    return found_seq, dp[:, -1].min(axis = 1).sum()#, dp