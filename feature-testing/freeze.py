### Example 3 : 



#   At the start, `param_2` is fixed
#   Update `param_1` during first 500 steps, then freeze it. 
#   Unfreeze `param_2` at 500th step, and optimize it till 1000th step



params = {
    'param_1': jnp.array([10.0]),
    'param_2' : jnp.array([10.0])
}

mask = {"param_1":"adam", "param_2":"zero" }

optimizer = optax.multi_transform(
                    {'adam': optax.adam(0.1), 
                     'zero': optax.set_to_zero()
                     },
                     mask)

opt_state = optimizer.init(params)

def compute_loss(params):
  return jnp.abs(params['param_1'][0]) + abs(params['param_2'][0])

for _ in range(1000):

    if(_ == 500):
        mask = {"param_1":"zero", "param_2":"adam" }
        optimizer = optax.multi_transform(
                    {'adam': optax.adam(0.1), 
                     'zero': optax.set_to_zero()
                     },
                     mask)

    #evaluate function's value and gradient
    loss, grads = jax.value_and_grad(compute_loss)(params)

    #obtain updates needed from gradients
    updates, opt_state = optimizer.update(grads, opt_state, params)

    #update parameters
    params = optax.apply_updates(params, updates)

    if(_%100==0):
        print("{:.5f}".format(loss,end = " \n"))