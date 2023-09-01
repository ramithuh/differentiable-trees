import math
import jax.numpy as jnp
from typing import Dict, List
from jaxtyping import Array, Float

def get_depth_from_node_index(
        node_index  : int,
        total_nodes : int
    ) -> int :
    """Given, the node ID, return it's depth in the tree

    Args:
        node_index: node ID (counting happens from leaves to root)
        total_nodes: total number of nodes in the tree

    Returns:
        depth of the node in the tree
    """
    depth = math.floor(math.log2(total_nodes - node_index))
    return depth

def get_max_node_id_for_depth(
        depth  : int,
        total_depth : int
    ) -> int :
    """Given, an arbitrary depth, return the maximum node id possible.
        (when depth is a larger value, returned id will be smaller)

    Args:
        depth : given depth,
        total : total_depth of the tree

    Returns:
        maximum node id possible
    """

    max_node_index = 2**(total_depth + 1) - 2**(depth)
    return max_node_index

def remove_branches(
        adj : Array,
        node : int,
    ) -> Array :
    """Given an adjacency matrix, recursively remove the descendents of the given `node`
    
    Args:
        adj : adjacency matrix,
        node : current node
    
    Returns:
        New adjacency matrix
    """
    children = jnp.where(adj[:,node] == 1)[0]
    
    for child in children:
        #remove current child
        adj = adj.at[child,node].set(0)

        #look at children of the chid
        adj = remove_branches(adj, child)
    
    return adj