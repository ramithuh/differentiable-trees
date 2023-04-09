import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from functools import partial
from typing import Dict, List
from jaxtyping import Array, Float

from modules.upgma import differentiable_upgma

@partial(jit, static_argnums=1)
def discretize_tree_topology(tree : Float[Array, "nodes nodes"], n_nodes : int) -> Float[Array, "nodes nodes"]:
    '''
        Returns the one-hot encoding of the given soft tree topology

        Args:
            tree : soft tree topology (softmax applied over the rows)
            n_nodes : number of total nodes in the tree
        Returns:
            t_q : one-hot encoding of the tree topology 
    '''

    t_argmax = jnp.argmax(tree,axis = 1)
    t_q = nn.one_hot(t_argmax, n_nodes)
    
    return t_q

@jit
def update_tree(params : Dict[str, Array], base_tree : Float[Array, "nodes nodes"]) -> Float[Array, "nodes nodes"]:
    '''
        Uses base_tree and returns the updated soft tree topology
        Particularly updates the ancestor probabilities in the trees

        Args:
            params : params['t'] denotes the trainable params; (ancestors)
                     params['t'] -> (n_nodes, n_ancestors)

            base_tree : base tree topology
                        base_tree -> (n_nodes, n_nodes)
        Returns:
            t : updated soft tree topology
                t -> (n_nodes, n_nodes)
    '''

    n_leaves = (base_tree.shape[1] - params['t'].shape[1])
    
    t_softmax = nn.softmax(params['t'],axis=1)

    t = base_tree.at[0:-1,n_leaves:].set(t_softmax)
    t = t.at[-1,-1].set(1) ### need to add this (so that the last sequence is matched with itself. i.e. root) 
    
    return t

@jit
def update_seq(params : Dict[str, Array], seqs : Float[Array, "nodes letters"], temperature : Float = 1 ) -> Float[Array, "nodes letters"]:
    '''
        Updates the ancestor sequences using the trainable params

        Args:
            params : params['n'] denotes the nth ancestor sequence
    '''


    n_all = seqs.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_all - n_leaves):
        key = str(i) #chr(97+i + n_leaves)
        
        seq      = nn.softmax(temperature*params[key])
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

@jit
def enforce_graph(t_ : Float[Array, "nodes nodes"], s : Float, metadata = None, verbose = False) -> List[Float]:
    ''' 

        Enforces constraints such that,
            1) the tree is bifurcating
            2) there are no self loops
        
        Args:
            t_: tree topology (after softmax)
            s : scaling factor
            verbose: print the loss values
        Returns:
            loss: loss value as list [tree_forcing_loss, loop_loss, bidirectional_loss]
    '''

    #n_all = metadata['n_all']
    n_all = t_.shape[0]
    n_leaves = (n_all + 1)//2
    n_ancestors = n_all - n_leaves

    # n_leaves = metadata['n_leaves']
    # n_ancestors = metadata['n_ancestors']

    tree_force_loss = jnp.sum(jnp.power(s*jnp.abs(jnp.sum(t_[:-1, n_all - n_ancestors: n_all], axis = 0) - 2),2))
    
    loop_loss = jnp.multiply(jnp.identity(n_all-1), t_[:-1,:-1]).sum()*100  ## penalize self connections
    
    bidirection_loss = jnp.sum(jnp.matmul(t_,t_)*jnp.identity(t_.shape[0]))*100 ## penalize bidirectional connections
    
    if(verbose):
        print("bifurcating tree_forcing_loss = ", tree_force_loss)
        print("self loop_loss = ", loop_loss)
        print("bidirectional loss = ", bidirection_loss)
    
    return tree_force_loss, loop_loss, bidirection_loss


def compute_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"], metadata, verbose = False, surrogate_loss = True) -> Float:
    '''
        compute the traversal cost (character level changes while traversing the tree)

        Args:
            sequences : one-hot encoded sequences
                        shape = (n_all, seq_length, letters)
            tree      : tree topology
                        shape = (n_all, n_all)
            
        Returns: 
            ans : traversal cost
    '''

    if(verbose==True):
        show_graph_with_labels(discretize_tree_topology(tree, n_all),n_leaves)
        
    seq_onehot = jnp.transpose(sequences, (2,0,1)) # bring onehot encoding to the front (onehot, n_all, seq_length)
    sel        = jnp.expand_dims(tree, axis = 0)   # make it (1, n_all, n_all)

    selection  = jnp.matmul(sel, seq_onehot)
    
    # print(selection)
    
    #method 1: (downside needs integers as we are indexing)
    #selection = selection.astype(int)
    #ans = sm[selection,sequences].sum()
    
    #method 2:
    if(surrogate_loss):
        '''
            This is an approximation of the traversal cost, because seq_onehot is soft 
        '''
        out = jnp.transpose(selection, (1,2,0))
        ans = (jnp.sum(abs((out - sequences)), axis = [1,2])/(2)).sum()

        #ans = ((nn.sigmoid((abs(selection - sequences))) - 0.5)*2).sum() ## works currently as sm is the basic config. need to think about extending
    else:
        '''
            This is the exact traversal cost, but it is not differentiable.
            similarity matrix is assumed to be a 1s matrix with 0s on the diagonal
        '''
        selection_ = jnp.matmul(discretize_tree_topology(tree, tree.shape[0]), jnp.argmax(sequences, axis = 2))
        sequences_ = jnp.argmax(sequences, axis = 2)
    
        sm  = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters'])
        
        ans = sm[jnp.round(selection_).astype(jnp.int32),jnp.round(sequences_).astype(jnp.int32)].sum()

    return ans

def DFS(curr : int, tree : Float[Array,"nodes nodes"], path_prob : float, visited : List[Float], depth : int, verbose=False ) -> List[Float]:
  '''
    Runs DFS on the tree and returns the visited nodes and the conflicts
    Args:
        curr    : current node
        tree    : tree topology matrix (adjacency matrix)
                  shape(n_all, n_all)
        path_prob : probability of the path till now
        visited   : list of visited nodes
        depth     : depth of the tree at the current node
        
        verbose : print the visited nodes
    Returns:
        visited : list of visited nodes
  '''

  if(verbose):
    print("-"*depth, "came to -> ", curr, " with path prob ", path_prob)

  ind = jnp.argpartition(tree[:-1,curr], -1)[-2:][::-1] #maxpool choice
  sel_ind = ind

  if(verbose):
    print("-"*depth, "🏃🏻‍♂️ going to visit -> ", sel_ind, " with probs : ", tree[:,curr][sel_ind])
  
  for j in range(0,sel_ind.shape[0]):
    i = sel_ind[j]

    #print("-"*depth, "visiting -> ", i, " with probs : ", lis[:,curr][i])

    if(visited[i] > 0): ## this is not general
      #conflicts[i] += tree[:,curr][i]*path_prob #*visited[i]    #scale by visited probability came form some node
      visited[i]    = visited[i] + tree[:,curr][i]*(path_prob) ##even if it's visited with a low probability, need to update
    else:
      visited[i]    = visited[i] + tree[:,curr][i]*(path_prob + 1e-10)
      visited = DFS(i, tree, tree[:,curr][i]*path_prob,  visited,  depth+1, verbose)

  if(verbose):
    print("-"*depth, "## came back to -> ", curr)
  
  return visited


def compute_cost_recursive(tree : Float[Array, "nodes nodes"], verbose=False) -> Float:
    """Computes the two combined costs of the tree
        1. loops (measured by conflicts)
        2. unvisited nodes

    Args:
        tree (shape = (n_all, n_all)): tree adjacency matrix
        verbose (bool, optional): print the visited nodes. Defaults to False.

    Returns:
        Float: loops + unvisited nodes
    """  

    visited = [0] * tree.shape[0]
    visited[-1] = 1

    visited = DFS(tree.shape[0] - 1, tree, 1, visited, 0, verbose = verbose)

    if(verbose):
        print("\n\n#######")
        print(" Visited nodes : ", jnp.asarray(visited))
        #print(" Conflicts : ", jnp.asarray(conflicts))
        print("#######\n\n")

        #print(params['t'].shape[0] - sum(visited),conflicts)

        print("\nDistance from prototype -> ", jnp.array(visited) - 1, " --> total ", abs(jnp.array(visited) - 1).sum() )
        # print("Total Conflicts --> ", jnp.array(conflicts).sum() )
    
    #print(params['t'].shape[0] - sum(visited)/, conflicts)
    return abs(jnp.array(visited) - 1).sum() #params['t'].shape[0] - sum(visited) +  conflicts


def compute_loss(params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], base_tree : Float[Array, "nodes nodes"], metadata : Dict, temp : Float, verbose = False) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    if(metadata['args']['fix_seqs']):
        seqs_ = seqs
    else:
        seqs_ = update_seq(params, seqs, temp )
    
    if(metadata['args']['fix_tree']):
        t_ = base_tree
    else:   
        t_ = differentiable_upgma(params['t'])#update_tree(params, base_tree)
    
    
    recurse_cost = 0 #compute_cost_recursive(t_, verbose = False)

    cost_surrogate = compute_cost(seqs_,t_, metadata) # surrogate traversal cost
    
    tree_force_loss, loop_loss, bidirection_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate + temp*(tree_force_loss + loop_loss + bidirection_loss + recurse_cost*500)

    if(verbose):
        cost           = compute_cost(seqs_,t_, metadata, surrogate_loss = False) # real traversal cost
        return cost, cost_surrogate, tree_force_loss, loop_loss, bidirection_loss, recurse_cost, loss

    return loss