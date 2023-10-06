import math

import jax
import jax.nn as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax.scipy.linalg as slin
import jax.numpy.linalg as la

from jax import jit
from functools import partial
from typing import Dict, List
from jaxtyping import Array, Float

from modules.upgma import differentiable_upgma
from modules.sankoff import run_diff_sankoff
from modules.sankoff import run_sankoff

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
def update_tree(params : Dict[str, Array], epoch : int = 0 ,  temp : Float = 1 ) -> Float[Array, "nodes nodes"]:
    '''
        Uses base_tree and returns the updated soft tree topology
        Particularly updates the ancestor probabilities in the trees

        Args:
            params : params['t'] denotes the trainable params; (ancestors)
                     params['t'] -> (n_nodes-1, n_ancestors)

            epoch : epoch number
            temp : temperature for softmax
            
        Returns:
            t : updated soft tree topology
                t -> (n_nodes, n_nodes)
    '''
    tree = params['t']#.sum(axis = 0)
    
    n_all = tree.shape[0] + 1 #(adding +1 because params doesn't have the root node)
    n_ancestors = tree.shape[1]
    n_leaves = n_all - n_ancestors
    
    key = jax.random.PRNGKey(epoch) # generate a random key
    gumbel_noise = jax.random.gumbel(key, (tree.shape[0], tree.shape[1])) # generate gumbel noise

    perturbed_tree_params = 1*tree + gumbel_noise*0.0 # add gumbel noise to the tree params

    inf_matrix = -jnp.ones((n_all, n_all))*jnp.inf
                    
    # generating the upper indices of the lower square matrix
    upper_indices = jnp.triu_indices(n_ancestors)
    lower_indices = jnp.tril_indices(n_ancestors)
    shifted_upper_indices = (upper_indices[0] + n_ancestors, upper_indices[1] + n_leaves )
    shifted_lower_indices = (lower_indices[0] + n_leaves, lower_indices[1] + n_leaves )

    #
    inf_matrix = inf_matrix.at[:n_ancestors, n_leaves:].set(perturbed_tree_params[:n_ancestors])
    inf_matrix = inf_matrix.at[shifted_upper_indices].set(perturbed_tree_params[n_ancestors:, :][upper_indices])
    
    # Lower part; comment unless using entropy loss
    #lower_triangle_values = 0 #perturbed_tree_params[n_ancestors:, :][lower_indices]
    #inf_matrix = inf_matrix.at[shifted_lower_indices].set(lower_triangle_values)
    
    inf_matrix = inf_matrix.at[-1,-1].set(1)
    
    return jax.nn.softmax(inf_matrix, axis = 1)


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
        
        # print(params[key].shape)
        # print(seq.shape)
        # break
        seq      = nn.softmax(params[key]*temperature)
        #hard_seq = (seq > 0.5).astype(int)
        
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
def entropy_loss(t_ : Float[Array, "nodes nodes"]) -> List[Float]:
    '''
        Pushes the nevada constraint on the relaxed adjacency matrix
    '''
    
    n_all = t_.shape[0]
    n_leaves = (n_all + 1)//2
    n_ancestors = n_all - n_leaves
    
    # generating the upper indices of the lower square matrix
    upper_indices = jnp.triu_indices(n_ancestors)
    lower_indices = jnp.tril_indices(n_ancestors)
    shifted_upper_indices = (upper_indices[0] + n_ancestors, upper_indices[1] + n_leaves )
    shifted_lower_indices = (lower_indices[0] + n_leaves, lower_indices[1] + n_leaves )
    
    blank = jnp.zeros((n_ancestors,n_ancestors))

    lower_triangle_values = t_[shifted_lower_indices]
    mat_1 = blank.at[lower_indices].set(lower_triangle_values)

    uppper_triangle_values = t_[shifted_upper_indices]
    mat_2 = blank.at[upper_indices].set(uppper_triangle_values)
    
    prob_vector = mat_2[1:].sum(axis = 1)

    cross_entropy_loss = -jnp.sum(1 * jnp.log(prob_vector))
    
    return cross_entropy_loss

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


    ancestor_columns = t_[:-1, n_all - n_ancestors: n_all]
    # scaling_vecotr = 

    tree_force_loss = jnp.sum(jnp.power(s*jnp.abs(jnp.sum(ancestor_columns, axis = 0) - 2),2))
    
    if(verbose):
        print("bifurcating tree_forcing_loss = ", tree_force_loss)

    return tree_force_loss

@jit
def compute_surrogate_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"]) -> Float:
    '''
    This is an approximation of the traversal cost, because seq_onehot is soft 
    '''
    
    seq_onehot = jnp.transpose(sequences, (2,0,1)) # bring onehot encoding to the front (onehot, n_all, seq_length)
    sel        = jnp.expand_dims(tree, axis = 0)   # make it (1, n_all, n_all)

    selection  = jnp.matmul(sel, seq_onehot)
    

    out = jnp.transpose(selection, (1,2,0))
    ans = (jnp.sum(abs((out - sequences)), axis = [1,2])/(2)).sum()
    
    return ans

@jit
def compute_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"], sm) -> Float:
    '''
        compute the traversal cost (character level changes while traversing the tree)
        This is the exact traversal cost, but it is not differentiable.
        similarity matrix is assumed to be a 1s matrix with 0s on the diagonal

        Args:
            sequences : one-hot encoded sequences
                        shape = (n_all, seq_length, letters)
            tree      : tree topology
                        shape = (n_all, n_all)
            
        Returns: 
            ans : traversal cost
    '''
    selection_ = jnp.matmul(discretize_tree_topology(tree, tree.shape[0]), jnp.argmax(sequences, axis = 2))
    sequences_ = jnp.argmax(sequences, axis = 2)
    
    ans = sm[jnp.round(selection_).astype(jnp.int64),jnp.round(sequences_).astype(jnp.int64)].sum()

    return ans

@jit
def leaf_to_root_connectivity_loss(
        adj : Array
    ) -> float:
    """Computes how many leaves are not connected to the root"""
    
    adj = adj.at[-1,-1].set(0) # additional check to remove self loop
    
    n_leaves = (adj.shape[0] + 1)//2
    
    depth_u_bound = math.ceil(math.log2(n_leaves))
    paths = jnp.zeros((n_leaves,))
    
    for path_length in range(1, depth_u_bound + 1):
        out = jnp.linalg.matrix_power(adj, path_length)
        
        paths += out[:n_leaves,-1]
    
    return n_leaves - paths.sum()

@jit
def compute_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, epoch : int) -> Float:
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
    
    t_ = update_tree(tree_params, epoch, temp)
    
    n_letters = 20
    cost_mat = jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters).astype(jnp.float64)
    _, cost_surrogate = run_diff_sankoff(t_, cost_mat, jnp.argmax(seqs, axis = 2), metadata, epsilon = (0.145e-1)/(1+epoch))
    #compute_surrogate_cost(seqs_,t_) # surrogate traversal cost

    tree_force_loss = enforce_graph(t_,10,metadata)
    
    cross_entropy_loss = 0 #entropy_loss(t_)
    connectivity_loss  = 0, #leaf_to_root_connectivity_loss(t_) 
    
    loss = cost_surrogate #+ temp*(tree_force_loss) # + cross_entropy_loss + connectivity_loss

    return loss


def compute_detailed_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, sm : Float[Array, "letters letters"], epoch : int = 0) -> Float:
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
    
    t_ = update_tree(tree_params, epoch, temp)
    
    n_letters = 20
    cost_mat = jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters).astype(jnp.float64)
    _, cost_surrogate = run_diff_sankoff(t_, cost_mat, jnp.argmax(seqs, axis = 2), metadata, epsilon = (0.145e-1)/(1+epoch))
    #compute_surrogate_cost(seqs_,t_) # surrogate traversal cost

    tree_force_loss = enforce_graph(t_,10,metadata)
    cross_entropy_loss = -1 #entropy_loss(t_) ##turned off
    connectivity_loss  = -1 #leaf_to_root_connectivity_loss(t_) 
    
    loss = cost_surrogate #+ temp*(tree_force_loss) #+ cross_entropy_loss + connectivity_loss

    _, _, cost = run_sankoff(discretize_tree_topology(t_, n_all), cost_mat, jnp.argmax(seqs, axis = 2), metadata, return_path = False)
        
    #compute_cost(seqs_,t_, sm) # real traversal cost

    return cost, cost_surrogate, tree_force_loss, cross_entropy_loss, connectivity_loss, loss