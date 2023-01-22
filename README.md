# differentiable-trees


Given leaf nodes, generate the most cost effective tree topology and ancestors using gradient descent.

Example experiment : https://wandb.ai/ramithx1/Differentiable-Trees/reports/Differentiable-Trees--VmlldzozMzg4MTE2?accessToken=cyle7be7zv37oxb05bdbsbqu1n3lyenrm69ly9y8swyf2s3emlt7vcy18z9zpgei

![ezgif-1-54b788b4ec](https://user-images.githubusercontent.com/3503098/213895875-989c9c49-cf7c-4f31-ab5c-68273e870c01.gif)

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


