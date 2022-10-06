import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def generate_t(n_nodes, seed = None):
  if(seed!=None):
    np.random.seed(seed)

  mat = np.zeros((n_nodes,n_nodes))

  order = np.arange(0,n_nodes-1,1)
  indices = order.copy()*0

  for i in range(0,n_nodes-1):
    id = np.random.choice(np.arange(i+1,n_nodes,1), 1)
    indices[i] = int(id)
    

  order = np.arange(0,n_nodes-1,1)

  #print(order,order.shape)
  #print(indices)


  mat[order,indices] = 1

  print(mat)
  return mat

def show_graph_with_labels(adjacency_matrix, label_names):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()#nx.DiGraph
    gr.add_edges_from(edges)
    
    color_map = ['red' if (node == 4 or node == 3) else 'yellow' for node in gr]        
    
    pos = graphviz_layout(gr, prog="dot")
    nx.draw(gr,pos, node_size=500 , labels = label_names,with_labels=True, node_color = color_map)
    plt.show()