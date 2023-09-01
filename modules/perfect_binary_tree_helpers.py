import math

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
