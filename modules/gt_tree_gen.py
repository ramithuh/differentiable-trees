import math
import jax
import jax.numpy as jnp
from jax import vmap
from jax import random
from jax import jit

from functools import partial

from modules.perfect_binary_tree_helpers import *
from modules.catalan_tools import *
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

  key_ = jax.random.split(key, num_mutations)

  mutations = vmap(jax.random.choice)(key_, options)

  return jnp.expand_dims(mutations, axis=1)

def create_seq(key, seq, depth):

  k1, k2 = jax.random.split(key)
  
  if(depth > target_depth):
    return

  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= jax.random.choice(k1, jnp.array(range(len(seq))), (num_mutations,1), replace=False)
  random_indexes[1]= jax.random.choice(k2, jnp.array(range(len(seq))), (num_mutations,1), replace=False)


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

    key = jax.random.PRNGKey(seed)

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

def traverse_and_create_seq(node, parent_seq, adj_matrix, key):
    global seq_dict
    
    ## if no children return dictionary;
    if(adj_matrix[:, node].sum() == 0):
        return

    assert adj_matrix[:, node].sum() == 2
    
    #print(f"traversing to {jnp.where(adj_matrix[:,node] == 1)[0]}, parent is {parent_seq}")
    
    ## Make 2 sequences for the two children:
    k1, k2 = jax.random.split(key)
    random_indexes = [[] for x in range(2)]
  
    random_indexes[0]= jax.random.choice(k1, jnp.array(range(len(parent_seq))), (num_mutations,1), replace=False)
    random_indexes[1]= jax.random.choice(k2, jnp.array(range(len(parent_seq))), (num_mutations,1), replace=False)

    seq_1 = parent_seq
    seq_2 = parent_seq

    seq_1 = seq_1.at[random_indexes[0]].set(mutate(k1, seq_1[random_indexes[0]]))
    seq_2 = seq_2.at[random_indexes[1]].set(mutate(k2, seq_2[random_indexes[1]]))
    
    ## Check if indeed x mutations were done to the parent
    assert (seq_1 != parent_seq).sum() == num_mutations
    assert (seq_2 != parent_seq).sum() == num_mutations
    
    ## If so, add them to the dictionary
    node_1 = jnp.where(adj_matrix[:,node] == 1)[0][0].item()
    node_2 = jnp.where(adj_matrix[:,node] == 1)[0][1].item()
        
    seq_dict[str(node_1)] = seq_1
    seq_dict[str(node_2)] = seq_2
    
    ## Traverse these 2 nodes, send them their new seq, and key
    traverse_and_create_seq(node_1, seq_1, adj_matrix, k1)
    traverse_and_create_seq(node_2, seq_2, adj_matrix, k2)

    return

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
    global seq_dict
    global num_mutations
    global sequences
    global n_letters
    global n_leaves
    
    metadata = metadata_.copy()
    
    num_mutations = metadata['n_mutations']
    n_all       = metadata['n_all']
    n_leaves    = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length  = metadata['seq_length']
    n_letters   = metadata['n_letters']

    if(perfect): 
      # perfect mode, possible only if N is a power of 2
      print_critical_info("Perfect tree mode")
      assert metadata['n_leaves'] == 2**int(math.log2(n))

      return generate_groundtruth(metadata, seed, verbose)
    
    seq_dict = {}

    seq = jnp.zeros(seq_length, dtype=jnp.int64)
    seq_dict[n_all-1] = seq

    tree_str  = generate_bifurcating_tree(seed, n_leaves)
    adj_matrix, _ = convert_parenthesis_notation_to_adj_matrix(tree_str)
    
    traverse_and_create_seq(n_all-1, seq, adj_matrix, jax.random.key(0))
    
    ### generate the outputs in the required form:
    sorted_keys = sorted(seq_dict.keys(), key=lambda x: int(x))

    # Use the sorted keys to get the corresponding arrays in order
    sorted_arrays = [seq_dict[key] for key in sorted_keys]

    # Stack the arrays along a new dimension
    true_main = jnp.stack(sorted_arrays)
    masked_main = true_main.at[n_leaves:n_all].set(0)
    
    adj_matrix = adj_matrix.at[-1,-1].set(1) ## manually fix root's self connection

    return masked_main, true_main, adj_matrix