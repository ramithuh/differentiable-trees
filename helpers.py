import jax
import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def generate_t(n_nodes, n_leaves, seed = None):
  assert n_nodes > n_leaves, "Hey! total nodes must be greater than leaves -_-"
    
  key = jax.random.PRNGKey(1701)
  if(seed!=None):
    key = jax.random.PRNGKey(seed)
  

  mat = jnp.zeros((n_nodes,n_nodes))

  order = jnp.arange(0,n_nodes-1,1)
  indices = order.copy()*0

  for i in range(0,n_nodes-1):
    key, subkey = jax.random.split(key)
    id = jax.random.choice(key,jnp.arange(max(n_leaves,i+1),n_nodes,1), [1])
    indices[i] = int(id)
    

  order = jnp.arange(0,n_nodes-1,1)

  #print(order,order.shape)
  #print(indices)

  mat = mat.at[order,indices].set(1)
  # mat[order,indices] = 1

  print(mat)
  return mat

def show_graph_with_labels(adjacency_matrix, n_leaves):
    label_names = {}
    
    for i in range(0,adjacency_matrix.shape[0]):
        label_names[i] = chr(97+i)
        
    
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()#nx.DiGraph
    gr.add_edges_from(edges)
    
    color_map = []
    for node in gr:
        if(node >= n_leaves):
            color_map.append('red')
        else:
            color_map.append('yellow')
        
    
    # color_map = ['red' if (node == 4 or node == 3) else 'yellow' for node in gr]        
    
    pos = graphviz_layout(gr, prog="dot")
    nx.draw(gr,pos, node_size=500 , labels = label_names,with_labels=True, node_color = color_map)
    plt.show()