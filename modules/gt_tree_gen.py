import math
import random
import jax.numpy as jnp

target_depth = 0
num_mutations = 0
sequences = []

def create_seq(seq, depth):
  print()
  

  if(depth > target_depth):
    return

  # index_1 = random.randint(0,len(seq)-1)
  # index_2 = random.randint(0,len(seq)-1)

  # seq_1 = seq[::]
  # seq_2 = seq[::]
  # seq_1[index_1] = int(not(seq[index_1]))
  # seq_2[index_2] = int(not(seq[index_2]))


  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= random.sample(range(len(seq)), num_mutations)
  random_indexes[1]= random.sample(range(len(seq)), num_mutations)


  seq_1 = seq[::]
  seq_2 = seq[::]
  for k in range(0,num_mutations):
    seq_1[random_indexes[0][k]] = int(not(seq[random_indexes[0][k]]))
    seq_2[random_indexes[1][k]] = int(not(seq[random_indexes[1][k]]))

  #seq
  print((depth+1)*'|--->',end="")
  print(seq_1)
  sequences[depth].append(seq_1)
  create_seq(seq_1, depth+1)
  

  #new_seq
  print((depth+1)*'|--->',end="")
  print(seq_2)
  sequences[depth].append(seq_2)
  create_seq(seq_2, depth+1)

def generate_groundtruth(metadata):
    global target_depth
    global num_mutations
    global sequences

    num_mutations = metadata['n_mutations']
    n_all = metadata['n_all']
    n_leaves = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length = metadata['seq_length']

    target_depth = int(math.log2(n_leaves))-1

    seq = [0 for x in range(seq_length)]

    sequences = [[] for x in range(target_depth + 1)]
    print(sequences)

    create_seq(seq,0)

    print(len(sequences[target_depth]))

    ### copy the leaves
    masked_main = sequences[target_depth].copy()
    true_main   = sequences[target_depth].copy()

    print(masked_main)


    n_leaves    = len(masked_main)
    n_ancestors = n_leaves - 1


    for i in range(0,n_ancestors):
        masked_main.append(seq)

    for i in range(0,len(sequences)-1):
        for k in range(0,len(sequences[i])):
            true_main.append(sequences[i][k])
    true_main.append(seq)

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

    return jnp.array(masked_main).astype(jnp.float32), jnp.array(true_main).astype(jnp.float32), jnp.array(tree).astype(jnp.float32)
