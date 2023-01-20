# differentiable-trees

```
conda create -n jax_env python=3.8
source activate jax_env
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=jax_env


#install remaining packages through pip
pip install -r requirements.txt
```

### Install JAX (fasrc cluster)
Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
```
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
