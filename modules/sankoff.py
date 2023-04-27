import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp

from typing import Dict, List
from jaxtyping import Array, Float

def run_dp(adj : Float[Array, "nodes nodes"], dp, dp_nodes, seq, cost_mat, args, verbose = False):
    '''
        Run sankoff algorithm given the tree topology and the leaf sequences
        
        Args:
            adj: adjacency matrix of the tree
            dp : blank dp table
            dp_nodes : blank dp table for storing the nodes
            seq : leaf sequences
            args : metadata
            verbose : print the progress of the algorithm
            
        Returns:
            dp : filled dp table
            dp_nodes : node information for backtracking
    '''
    
    
    n_leaves = args['n_leaves']
    n_all    = args['n_all']
    n_letters = args['n_letters']

    for i in range(0, n_leaves): # initailize the dp table
        dp =  dp.at[i,seq[i].astype(int)].set(0)


    for node in range(n_leaves, n_all):
        ind = jnp.argpartition(adj[:,node], -1)[-2:]

        left_node  = ind[0]
        right_node = ind[1]

        if(verbose):
            print(f"at node {node+1} children are : {left_node+1} and {right_node+1}")

        for i in range(n_letters):
            left_cost_array = cost_mat[i][::] + dp[left_node][::]
            left_cost = jnp.min(left_cost_array)
            left_char = jnp.argmin(left_cost_array)

            right_cost_array = cost_mat[i][::] + dp[right_node][::]
            right_cost = jnp.min(right_cost_array)
            right_char = jnp.argmin(right_cost_array)

            dp = dp.at[node,i].set(left_cost + right_cost)

            dp_nodes = dp_nodes.at[node, i, 0].set(left_node)
            dp_nodes = dp_nodes.at[node, i, 1].set(left_char)

            dp_nodes = dp_nodes.at[node, i, 2].set(right_node)
            dp_nodes = dp_nodes.at[node, i, 3].set(right_char)

            # dp_nodes[...,node][i] = {str(left_node) : left_char, str(right_node) : right_char}

    return dp, dp_nodes

vectorized_dp = jax.vmap(run_dp, (None, 0, 0, 1, None, None, None), 0)

def backtrack_dp(node, letter, seq_chars, connections, args):
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

    if(node < args['n_leaves']): #break if it's a leaf
        return seq_chars


    seq_chars = seq_chars.at[node].set(letter)

    child = connections[node,letter]

    seq_chars = backtrack_dp(child[0].astype(int), child[1].astype(int), seq_chars, connections, args)
    seq_chars = backtrack_dp(child[2].astype(int), child[3].astype(int), seq_chars, connections, args)

    return seq_chars

def run_sankoff(adj, cost_mat, seq, metadata):
    adj = adj.at[-1,-1].set(0) ##manually remove self connection (convention used in the train script)
    
    ##empty dp tables
    dp_nodes = jnp.zeros((seq.shape[1], metadata['n_all'], metadata['n_letters'], 4))
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], metadata['n_letters']))*1e7
    
    
    dp, connections = vectorized_dp(adj, dp, dp_nodes, seq, cost_mat, metadata, False)

    
    seq_chars = jnp.zeros((seq.shape[1], metadata['n_all'],1)) 
    
    node = adj.shape[0] - 1

    for i in range(0,seq.shape[1]):
        letter = jnp.argmin(dp[i,node,:])

        chars = backtrack_dp(node, letter, seq_chars[i], connections[i], metadata).reshape(metadata['n_all'],)

        seq = seq.at[metadata['n_leaves']:,i].set(chars[metadata['n_leaves']:])
    
    return seq, dp[:, -1].min(axis = 1).sum()#, dp