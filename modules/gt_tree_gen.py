import math
import jax
import jax.numpy as jnp
from jax import vmap
from jax import random
from jax import jit

from functools import partial

from modules.perfect_binary_tree_helpers import *
from jaxtyping import Array, Float
from modules.vis_utils import *
from typing import Dict, List
from arg_parser_v2 import *

target_depth = 0
num_mutations = 0
sequences = []
n_letters = 0

def mutate(key, exclude_indexes):
  '''
    Mutates a sequence by randomly changing one of the letters

    Accepts:
      exclude_indexes: list of indexes to exclude from the mutation (shape = (n_mutations,1))

    Returns:
      mutated sequence
  '''
  global n_letters

  space = jnp.tile(jnp.arange(n_letters), (num_mutations, 1)) 
  options  = space[space != exclude_indexes].reshape((space.shape[0],space.shape[1] - exclude_indexes.shape[1]))

  key_ = random.split(key, num_mutations)

  mutations = vmap(random.choice)(key_, options)

  return jnp.expand_dims(mutations, axis=1)

def create_seq(key, seq, depth):

  k1, k2 = random.split(key)
  
  if(depth > target_depth):
    return

  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= random.choice(k1, jnp.array(range(len(seq))), (num_mutations,1), replace=False)
  random_indexes[1]= random.choice(k2, jnp.array(range(len(seq))), (num_mutations,1), replace=False)


  seq_1 = seq
  seq_2 = seq

  seq_1 = seq_1.at[random_indexes[0]].set(mutate(k1, seq[random_indexes[0]]))
  seq_2 = seq_2.at[random_indexes[1]].set(mutate(k2, seq[random_indexes[1]]))


  #seq
  sequences[depth].append(seq_1)
  create_seq(k1, seq_1, depth+1)
  
  #new_seq
  sequences[depth].append(seq_2)
  create_seq(k2, seq_2, depth+1)

def generate_groundtruth(metadata : Dict[str, int], seed = 42, verbose = False) -> List[Array]: 
    '''
        Generates a groundtruth example based on the metadata provided


        Args : 
            metadata: dictionary containing the required specifications
            seed: random seed
            verbose: print the number of leaves generated

        Returns:
            masked_main : masked sequences (shape = (n_all, seq_length))
            true_main   : true sequences   (shape = (n_all, seq_length))
            tree        : tree structure   (shape = (n_all, n_all))
    '''

    key = random.PRNGKey(seed)

    global target_depth
    global num_mutations
    global sequences
    global n_letters

    num_mutations = metadata['n_mutations']
    n_all       = metadata['n_all']
    n_leaves    = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length  = metadata['seq_length']
    n_letters   = metadata['n_letters']

    target_depth = int(math.log2(n_leaves))-1

    seq = jnp.zeros(seq_length, dtype=jnp.int64)

    sequences = [[] for x in range(target_depth + 1)]
    create_seq(key,seq,0)

    if(verbose):
      print(len(sequences[target_depth]))

    ### copy the leaves
    masked_main = sequences[target_depth].copy()
    true_main   = sequences[target_depth].copy()

    n_leaves    = len(masked_main)
    n_ancestors = n_leaves - 1


    for i in range(0,n_ancestors):
        masked_main.append(seq)

    ## Generate the true sequences by traversing through the leaves to the root
    for i in range(len(sequences)-2, -1, -1):
        for k in range(0,len(sequences[i])):
            true_main.append(sequences[i][k])
    true_main.append(seq)

    if(verbose):
      i = 0
      for item in true_main:
        print(">seq"+str(i))
        for char in item:
          print(char, end="")
        print()

        i+=1

    tree = []

    iter = n_ancestors

    for i in range(0,len(masked_main)):
        leave = [0]*len(masked_main)

        if(i%2==0 and i!=len(masked_main)-1):
            iter +=1 

        leave[iter] = 1
        tree.append(leave)

    return jnp.array(masked_main).astype(jnp.bfloat16), jnp.array(true_main).astype(jnp.bfloat16), jnp.array(tree).astype(jnp.bfloat16)

def update_probability(
        depth_bounds : List[int],
        ignored_nodes : Array,
        curr_prob_vector : Array
    ) -> Array:
    """Updates the probability array based on two things. 1) The depth bounds and 2) The mask

    Args:
        depth_bounds: list of depth bounds (upper and lower)
        ignored_nodes : list of nodes already removed or ignored (probabilities will be made zero)
        curr_prob_vector: current probabilities of the nodes in the tree

    Returns:
        updated probability vector
    """
    depth_u_bound = depth_bounds[1] # max depth of the perfect binary tree

    u_bound = get_max_node_id_for_depth(depth_bounds[0], depth_u_bound)
    l_bound = get_max_node_id_for_depth(depth_bounds[0]+1, depth_u_bound)

    curr_prob_vector = curr_prob_vector.at[0:u_bound].set(1)

    if(ignored_nodes.shape[0] > 0): # There are some nodes that we need to remove from selecting
        curr_prob_vector = curr_prob_vector.at[ignored_nodes].set(0)

    return curr_prob_vector

def extract_key_parts(adj):
    """Extracts key elements which make up a tree
    
    Args:
        adj: adjacency matrix of the tree
    
    Returns:
        leaf_ids: ids of the leaves
        ance_ids: ids of the ancestors
        orphans_: ids of the orphan nodes (which can be removed)
    """
    temp_adj = adj.at[-1,-1].set(0) # temporary copy
    
    leaf_ids = find_children_ids(temp_adj, temp_adj.shape[0] - 1)
    leaf_ids = jnp.array([x.item() for x in leaf_ids]).sort()
    
    ance_ids = jnp.where(temp_adj.sum(axis = 0) == 2)[0].sort()
    orphans_ = jnp.where(adj.sum(axis = 1) == 0)[0]
    
    return leaf_ids, ance_ids, orphans_  

def generate_general_groundtruth(
    metadata_ : Dict[str, int], 
    perfect : bool = False, 
    seed : int = 42, 
    verbose : bool = False,
    show_derivation : bool = False
    ):
    """Generates a random groundtruth example based on provided specs

    Args : 
        metadata:
            * number of leaves (N)
            * number of mutations (m)
            * number of letters (c)
            * sequence length (l)

        perfect : if True, the tree is perfect (all leaves have the same depth)
        seed : random seed
        verbose: print the number of leaves generated

    Returns:
        masked_main : masked sequences (shape = (n_all, seq_length)) -> leaves visible, ancestors masked
        true_main   : true sequences   (shape = (n_all, seq_length)) -> leaves & ancestors visible
        tree        : tree structure   (shape = (n_all, n_all)) 	 -> ground truth tree
    """
    metadata = metadata_.copy()
    n = metadata['n_leaves']

    if(perfect): 
        # perfect mode, possible only if N is a power of 2
        print_critical_info("Perfect tree mode")
        assert metadata['n_leaves'] == 2**int(math.log2(n))
        
        return generate_groundtruth(metadata, seed, verbose)
    else:
        # Compute params for the prospective perfect tree (upper bound for depth, how many to remove)
        depth_u_bound = math.ceil(math.log2(n))
        leaf_u_bound  = 2**int(depth_u_bound)
        leaves_to_remove     = leaf_u_bound - n
        metadata['n_leaves'] = leaf_u_bound
        
        # Generate the perfect tree
        print_bold_info(f"Generating perfect tree of {leaf_u_bound} leaves in order to make tree with {n} leaves, seed = {seed}")
        leaf_seqs, all_seqs, adj = generate_groundtruth(metadata, seed, verbose)

        # Initialize depth bounds
        # Based on how many to remove, which depth can we go as the most top level ?
        depth_bounds = [None, depth_u_bound]
        depth_bounds[0] = math.ceil(depth_u_bound - math.log2(leaves_to_remove+1)) 

        # Initialize params
        done = False if leaves_to_remove > 0 else True
        prob = jnp.zeros(leaf_u_bound + leaf_u_bound - 1)  # Probability Array (2*n-1 total nodes)
        nodes = jnp.array(range(leaf_u_bound + leaf_u_bound - 1))
        
        # Ignore all leaves (no point of removing them, we can remove parents)
        ignored_nodes = jnp.arange(leaf_u_bound)

        key = jax.random.PRNGKey(seed)
        
        k = 0
        while(not done):
            # Obtain valid updated probabilities
            prob = jnp.zeros(leaf_u_bound + leaf_u_bound - 1)
            prob = update_probability(depth_bounds, ignored_nodes, prob) 
            
            # Select an ancestor, to chop its both branches
            sel = random.choice(key, nodes, p=prob, axis=0)
            ## remove connections of these nodes from the tree
            adj, removed_leaf_count, nodes_in_path = remove_branches(adj, sel)
            
            # Keep track of all removed nodes (past, selected parent, nodes affected)
            ignored_nodes = jnp.concatenate([ignored_nodes, jnp.array([sel]), jnp.array(nodes_in_path)])
            
            # Update next bounds to pick 
            leaves_to_remove = leaves_to_remove - (removed_leaf_count - 1)
            depth_bounds[0] = math.ceil(depth_u_bound - math.log2(leaves_to_remove+1)) 
          
            if(leaves_to_remove == 0):
                done = True
                
            k+=1
            key, _ = random.split(key)
    
    ## Extract indices of key stuff
    leaf_ids, ance_ids, orphans_ = extract_key_parts(adj)
    
    ## Extract leaves and ancestors from the original seq matrix
    extracted_leaves = all_seqs[leaf_ids]
    extracted_ancestors = all_seqs[ance_ids]
    
    ## Checking whether groundtruth generation worked correctly
    assert n == extracted_leaves.shape[0]
    
    dummy_ancestors = jnp.zeros((n-1, extracted_leaves.shape[1]))
    new_leaves = jnp.vstack((extracted_leaves, dummy_ancestors))
    
    ## Cleanup the pruned tree by removing orphan nodes
    shrinked_adj = adj.copy()
    shrinked_adj = jnp.delete(shrinked_adj, orphans_, axis = 0)
    shrinked_adj = jnp.delete(shrinked_adj, orphans_, axis = 1)
    
    ## Extract indices of key stuff
    new_leaf_ids, new_ance_ids, new_orphans_ = extract_key_parts(shrinked_adj)
    
    ## Build a new seqs matrix with correct ordering which matches shrinked_adj
    new_all_seqs = jnp.zeros((new_leaves.shape[0], new_leaves.shape[1]))
    new_all_seqs = new_all_seqs.at[new_leaf_ids].set(extracted_leaves)
    new_all_seqs = new_all_seqs.at[new_ance_ids].set(extracted_ancestors)
    
 
    if(show_derivation):
        '''
            For debugging purposes (displaying how perfect tree was pruned)
        '''
        
        print(all_seqs)
        print_critical_info(f"-> extracting new leaf ids : {leaf_ids}")
        print_critical_info(f"-> extracting remaining ancestor ids : {ance_ids}")
        
        print_success_info(f"Extracted Leaves")
        print_success_info(f"\n{extracted_leaves}")
        
        print_success_info(f"Constructed New Seq Matrix")
        print_success_info(f"\n{new_all_seqs}")
        
        orphans = jnp.where(adj.sum(axis = 1) == 0)[0]

        label_names = {}
        for i in range(0, adj.shape[0]):
            if i in orphans:
                continue
            label_names[i] = str(i)
        
        # plot original numbering, displaying what's left
        fig = show_graph_with_labels(adj, n, [label_names, leaf_ids], True)
        
    
        label_names  = {}
        for i in range(0, shrinked_adj.shape[0]):
            label_names[i] = str(i) 
        
        return [new_leaves, new_leaf_ids], [new_all_seqs, label_names], [shrinked_adj, adj, fig]

    ##            
    return new_leaves, new_all_seqs, shrinked_adj