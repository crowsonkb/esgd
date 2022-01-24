# ESGD-M

ESGD-M is a stochastic non-convex second order optimizer, suitable for training deep learning models. It is based on ESGD ([Equilibrated adaptive learning rates for non-convex optimization](https://proceedings.neurips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf)) and incorporates quasi-hyperbolic momentum ([Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)) to accelerate convergence, which considerably improves its performance over plain ESGD.

ESGD-M obtains Hessian information through occasional Hessian-vector products (by default, every ten optimizer steps; each Hessian-vector product is approximately the same cost as a gradient evaluation) and uses it to adapt per-parameter learning rates. It estimates the diagonal of the absolute Hessian, diag(|H|), to use as a diagonal preconditioner.

To use this optimizer you must call `.backward()` with the `create_graph=True` option on those steps when it is going to perform a Hessian-vector product. You can call it like: `loss.backward(create_graph=opt.should_create_graph())` to do this. Gradient accumulation steps and distributed training are currently not supported.

## Learning rates

ESGD-M learning rates have a different meaning from SGD and Adagrad/Adam/etc. You may need to try learning rates in the range 1e-3 to 1.

SGD class optimizers:

* If you rescale your parameters by a factor of n, you must scale your learning rate by a factor of n^2.

* If you rescale your loss by a factor of n, you must scale your learning rate by a factor of 1 / n.

Adagrad/Adam class optimizers:

* If you rescale your parameters by a factor of n, you must scale your learning rate by a factor of n.

* If you rescale your loss by a factor of n, you do not have to scale your learning rate.

Second order optimizers (including ESGD-M):

* You do not have to scale your learning rate if you rescale either your parameters or your loss.

## Momentum

The default configuration is Nesterov momentum (if `v` is not specified then it will default to the value of `beta_1`, producing Nesterov momentum):

```python
opt = ESGD(model.parameters(), lr=1, betas=(0.9, 0.999), v=0.9)
```

The Quasi-Hyperbolic Momentum recommended defaults can be obtained using:

```python
opt = ESGD(model.parameters(), lr=1, betas=(0.999, 0.999), v=0.7)
```

Setting `v` equal to 1 will do normal (non-Nesterov) momentum.

The ESGD-M decay coefficient `beta_2` refers not to the squared gradient as in Adam but to the squared Hessian diagonal estimate, which it uses in place of the squared gradient to provide per-parameter adaptive learning rates.

## Hessian-vector products

The absolute Hessian diagonal diag(|H|) is estimated every `update_d_every` steps. The default is 10. Also, for the first `d_warmup` steps the diagonal will be estimated regardless, to obtain a lower variance estimate of diag(|H|) quickly. The estimation uses a Hessian-vector product, which takes around the same amount of time as a gradient evaluation to compute. You must explicitly signal to PyTorch that you want to do a double backward pass  on the steps when the optimizer is scheduled to do it by:

```python
opt.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward(create_graph=opt.should_create_graph())
opt.step()
```

## Weight decay

Weight decay is performed separately from the Hessian-vector product and the preconditioner, similar to AdamW except that the weight decay value provided by the user is multiplied by the current learning rate to determine the factor to decay the weights by.

## Learning rate warmup

Because the diag(|H|) estimates are high variance, the adaptive learning rates are not very reliable before many steps have been taken and many estimates have been averaged together. To deal with this ESGD-M has a short exponential learning rate warmup by default (it is combined with any external learning rate schedulers). On each step (starting from 1) the learning rate will be:

`lr * (1 - lr_warmup**step)`

The default value for `lr_warmup` is 0.99, which reaches 63% of the specified learning rate in 100 steps and 95% in 300 steps.
