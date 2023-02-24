import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from functools import partial


@partial(jit, static_argnums=1)
def discretize_tree_topology(tree, n_nodes):
    t_argmax = jnp.argmax(tree,axis = 1)
    t_q = nn.one_hot(t_argmax, n_nodes)
    
    return t_q

@jit
def update_tree(params, base_tree):
    n_leaves = (base_tree.shape[1] - params['t'].shape[1])
    
    t_softmax = nn.softmax(params['t'],axis=1)

    t = base_tree.at[0:-1,n_leaves:].set(t_softmax)
    t = t.at[-1,-1].set(1) ### need to add this (so that the last sequence is matched with itself) 
    
    return t

@jit
def update_seq(params, seqs, temperature = 1):


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


def compute_cost(sequences, tree, metadata, verbose = False, surrogate_loss = True):
    if(verbose==True):
        show_graph_with_labels(discretize_tree_topology(tree, n_all),n_leaves)
        
    seq_onehot = jnp.transpose(sequences, (2,0,1)) #bring onehot encoding to the front
    sel        = jnp.expand_dims(tree, axis = 0)   # make it (1, n_all, n_all)

    selection = jnp.matmul(sel, seq_onehot)

    selection_ = jnp.matmul(tree, jnp.argmax(sequences, axis = 2))
    sequences_ = jnp.argmax(sequences, axis = 2)
    
    # print(selection)
    
    #method 1: (downside needs integers as we are indexing)
    #selection = selection.astype(int)
    #ans = sm[selection,sequences].sum()
    
    #method 2:
    if(surrogate_loss):
        out = jnp.transpose(selection, (1,2,0))
        ans = (jnp.sum(abs((out - sequences)), axis = [1,2])/(2)).sum()

        #ans = ((nn.sigmoid((abs(selection - sequences))) - 0.5)*2).sum() ## works currently as sm is the basic config. need to think about extending
    else:
        sm = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters'])
        ans = sm[jnp.round(selection_).astype(jnp.int32),jnp.round(sequences_).astype(jnp.int32)].sum()

    return ans

def DFS(curr, tree, path_prob, visited,depth, verbose=False ):
  '''
    Runs DFS on the tree and returns the visited nodes and the conflicts
    Args:
        curr: current node
        tree: tree topology matrix (adjacency matrix)
        visited: list of visited nodes
        conflicts: number of conflicts
        verbose: print the visited nodes
    Returns:
        visited: list of visited nodes
        conflicts: number of conflicts
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


def compute_cost_recursive(tree, verbose=False):
  '''
    Computes the two costs of the tree
        loops (measured by conflicts)
        unvisited nodes

    Args:
        tree: tree adjacency matrix
        verbose: print the visited nodes

    Returns:
        loops + unvisited nodes
  '''

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


def compute_loss(params, seqs, base_tree, metadata, temp, verbose = False):
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    if(metadata['args']['fix_seqs']):
        seqs_ = seqs
    else:
        seqs_ = update_seq(params, seqs, temp )
    
    if(metadata['args']['fix_tree']):
        t_ = base_tree
    else:
        t_ = update_tree(params, base_tree)
    
    
    recurse_cost = -1 #compute_cost_recursive(t_, verbose = False)

    cost_surrogate = compute_cost(seqs_,t_, metadata) # surrogate traversal cost
    
    tree_force_loss, loop_loss, bidirection_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate + temp*(tree_force_loss + loop_loss + bidirection_loss + recurse_cost*0)

    if(verbose):
        cost           = compute_cost(seqs_,t_, metadata, surrogate_loss = False) # real traversal cost
        return cost, cost_surrogate, tree_force_loss, loop_loss, bidirection_loss, recurse_cost, loss

    return loss