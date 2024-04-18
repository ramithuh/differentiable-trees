import subprocess
import os
import math
import jax
        
print("Which GPU do you want to restrict to?")
print(jax.devices('gpu'))
id = input()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

print("how much percentage to dedicate from gpu (eg. .90)?")
percentage = input()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = percentage

print("Any notes for this batch run?")
note = input()

# Define the list of seeds you want to use
seeds = [x+42 for x in range(10)]#[::-1]

# Define the leaf complexity you want to use
print("which leaf complexity do you want to use?")
print("start value: ?")
s = int(input())
print("end value: ?")
e = int(input())

# leaves = [2**x for x in range(int(math.log2(s)), int(math.log2(e))+1)]
leaves = []
curr = s
while(curr <= e):
    leaves.append(curr)
    curr = curr*2

print("ok will run for the following leaf complexities:")
print(leaves)

# Common command and arguments
base = ["python", "train_batch_implicit_diff.py", "-nl", "20", "-m", "50", "-sl", "256", "-tLs", "[0,0.005,10,50]", "-lr", "0.1","-lr_seq", "0.01","-t", "diverse_trees", "-p","Batch-Run-Maximum-Parsimony-A100-META-optimized", "-alt", "-n", note]
base = base + ["-g", "0"]

for l in leaves:
    cmd = base + ["-l", str(l)]
    
    if(l < 32):
        cmd = cmd + ["-e", "5000"]
        cmd = cmd + ["-ai", "1", "-ic", "100"]
    elif(l < 64):
        cmd = cmd + ["-e", "9000"]
        cmd = cmd + ["-ai", "2", "-ic", "100"]
    elif(l <= 128):
        cmd = cmd + ["-e", "9000"]
        cmd = cmd + ["-ai", "3", "-ic", "10"]
    else:
        cmd = cmd + ["-e", "9000"]
        cmd = cmd + ["-ai", "4", "-ic", "10"]
    
    print(f"Running for {l} leaves")
    
    for seed in seeds:
        # Add the seed argument to the command
        cmd_with_seed = cmd + ["-s", str(seed)]

        # Run the command with the modified seed
        print(f"-> Running for seed {seed}: ", end =", ")
        
        #Use Popen instead of run and store the process object
        process = subprocess.Popen(cmd_with_seed, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        #Print the PID
        print(f"PID: {process.pid}")
        print(f"{process.stderr.read()}")
        
        process.wait()
