import networkx as nx
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation
from netgraph import Graph
from networkx.drawing.nx_pydot import graphviz_layout
import math
import jax.numpy as jnp
import jax
import numpy as np
import random

char_map = {'0': '(',
       '1': ')'
      }

def catalan(n):
    if (n == 0 or n == 1):
        return 1
 
    # Table to store results of subproblems
    catalan = [0 for i in range(n + 1)]
 
    # Initialize first two values in table
    catalan[0] = 1
    catalan[1] = 1
 
    # Fill entries in catalan[] using recursive formula
    for i in range(2, n + 1):
        catalan[i] = 0
        for j in range(i):
            catalan[i] += catalan[j] * catalan[i-j-1]
 
    # Return last entry
    return catalan[n]

def convert_parenthesis_notation_to_adj_matrix(tree_str):
  """
    Given a parenthesis string, generate the `adj_list` and `adj_matrix`;
    where the adj_matrix will be ordered; (leaves are given 0-n)

  """
  n_nodes = (len(tree_str)+1) # total number of nodes

  # node number we are currently at (left, bottom cell); we keep decreasing
  curr_node = n_nodes

  ## keep track of upcoming parents available, that we can utilize to add another child to the right
  parents_queue = [curr_node]

  adj_list = {}
  leaves = []

  for j in range(0,len(tree_str)): # loop through the string
    pos = tree_str[j]
    
    # curr_node ID
    curr_node -=1

    if(pos == "("):
      # we are essentially adding another parent candidate
      parent = parents_queue[-1]
    elif(pos == ")"):
      # we are essentially removing the chance of the poential parent being encountered again
      leaf = parents_queue[-1] #we can no longer add children to this node, so it's a leaf.
      leaves.append(leaf)
      
      parents_queue.pop()
      parent = parents_queue[-1]
      parents_queue.pop() # we added a node to the right, so it cant be a parent no longer; thus, pop!

    if(str(parent) not in adj_list.keys()):
      adj_list[str(parent)] = []

    adj_list[str(parent)].append(curr_node)
    parents_queue.append(curr_node)
  
  leaves.append(curr_node)

  ## convert adjacency list to matrix
  adj_matrix = jnp.zeros((n_nodes, n_nodes))
  
  ## Create a mapping to rename ancestors and leaves to a different ordering; leaves are given priority 0 to n.
  mapping = {}
  for l_id, leaf in enumerate(leaves):
    mapping[str(leaf)] = l_id #real lead id (reordered)
    
  ## Give rest of the ordering to the ancestors
  for p_id, key in enumerate(sorted([int(x) for x in adj_list.keys()])[::-1]):
    mapping[str(key)] = n_nodes - 1 - p_id #real parent id

  ## loop through the adj_list;
  for key in adj_list.keys():    
    for child in adj_list[key]:
      adj_matrix = adj_matrix.at[mapping[str(child)], mapping[key]].set(1)
  

  ## Sanity check to see if it's topologically sorted.
  lower_indices = jnp.tril_indices(adj_matrix.shape[0])
  lower_triangle_sum = adj_matrix[lower_indices].sum()
  assert lower_triangle_sum == 0
        
  return adj_matrix, adj_list #(unordered adj_list)

def generate_bifurcating_tree(seed,n_leaves):
  """
   Dyck word (path) based tree generation
  """

  key1 = jax.random.key(seed)

  total_nodes = n_leaves*2 - 1
  total_edges = total_nodes - 1

  tree_str = ""

  l_count = 0
  r_count = 0

  iter = 0
  while(len(tree_str) < total_edges):
    ## loop until we have added all the edges

    val = jax.random.randint(key1, minval=0, maxval=2, shape= (1,))[0].item()
    key1, key2 = jax.random.split(key1)

    r_outcome = r_count + val
    l_outcome = l_count + (1 if val == 0 else 0)

    ## right paranthesis cannot exceed left ones;
    if(r_outcome > l_outcome or l_outcome > total_edges//2 or r_outcome > total_edges//2 ):
      continue
    else:
      tree_str += char_map[str(val)]
      r_count += val
      l_count += (1 if val == 0 else 0)

  return tree_str

def generate_random_tree_distribution(n_leaves, count, verbose = False):
    """
    Given the number of leaves needed, generate `count` number of random trees, save in hashmap
    
    Returns: 
        Return statistics of the tree distribution
    """

    hash_map = {}

    for seed in range(0,count):
        
        tree_str  = generate_bifurcating_tree(seed, n_leaves)
       
        if(verbose):
            print(tree_str)
            adj_matrix, adj_list = convert_parenthesis_notation_to_adj_matrix(tree_str)
            fig = show_graph_with_labels(adj_matrix, n_leaves, return_img = True)
        
        if(tree_str in hash_map.keys()):
            hash_map[tree_str] +=1
        else:
            hash_map[tree_str] = 1
        
    return hash_map