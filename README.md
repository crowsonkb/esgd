# ESGD-M

ESGD-M is a stochastic non-convex second order optimizer, suitable for training deep learning models. It is based on ESGD ([Equilibrated adaptive learning rates for non-convex optimization](https://arxiv.org/abs/1502.04390)) and incorporates Nesterov momentum to accelerate convergence, which considerably improves its performance over plain ESGD. The absolute Hessian diagonal estimate is also decayed with an [Adamax](https://arxiv.org/abs/1412.6980) style exponentially weighted inf norm, instead of a simple average as in the original ESGD.

ESGD-M obtains Hessian information through occasional Hessian-vector products (by default, every ten optimizer steps; each Hessian-vector product is approximately the same cost as a gradient evaluation) and uses it to adapt per-parameter learning rates.

To use this optimizer you must call `.backward()` with the `create_graph=True` option on those steps when it is going to perform a Hessian-vector product. You can call it like: `loss.backward(create_graph=opt.should_create_graph())` to do this. Gradient accumulation steps and distributed training are currently not supported.

## Difference between versions

This is an updated version of ESGD-M that is more stable (diverges and produces NaN less). It uses the exponentially weighted inf norm (Adamax) for the absolute Hessian diagonal estimate instead of an EMA, in order to make the optimizer go slower along a coordinate when it is less certain about the correct value of the absolute Hessian diagonal for that coordinate. Also, quasi-hyperbolic momentum was taken out because nobody used it. Nesterov momentum is the only option now. The learning rate warmup is also slower now because only steps where there was a Hessian-vector product performed count for it.

## Learning rates

ESGD-M learning rates have a different meaning from SGD and Adagrad/Adam/etc. You may need to try learning rates in the range 1e-3 to 1, starting around 0.5.

SGD class optimizers:

* If you rescale your parameters by a factor of n, you must scale your learning rate by a factor of n^2.

* If you rescale your loss by a factor of n, you must scale your learning rate by a factor of 1 / n.

Adagrad/Adam class optimizers:

* If you rescale your parameters by a factor of n, you must scale your learning rate by a factor of n.

* If you rescale your loss by a factor of n, you do not have to scale your learning rate.

Second order optimizers (including ESGD-M):

* You do not have to scale your learning rate if you rescale either your parameters or your loss.

## Hessian-vector products

The equilibriation preconditioner sqrt(diag(H^2)) is estimated every `update_d_every` steps. The default is 10. Also, for the first `d_warmup` steps the diagonal will be estimated regardless, to obtain a lower variance estimate of sqrt(diag(H^2)) quickly. The estimation uses a Hessian-vector product, which takes around the same amount of time as a gradient evaluation to compute. You must explicitly signal to PyTorch that you want to do a double backward pass on the steps when the optimizer is scheduled to do it by:

```python
opt.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward(create_graph=opt.should_create_graph())
opt.step()
```

## Weight decay

Weight decay is performed separately from the Hessian-vector product and the preconditioner, similar to AdamW except that the weight decay value provided by the user is multiplied by the current learning rate to determine the factor to decay the weights by.

## Learning rate warmup

Because the sqrt(diag(H^2)) estimates are high variance, the adaptive learning rates are not very reliable before many steps have been taken and many estimates have been averaged together. To deal with this ESGD-M has a short exponential learning rate warmup by default (it is combined with any external learning rate schedulers). On each step (starting from 1) the learning rate will be:

`lr * (1 - lr_warmup**step)`

The default value for `lr_warmup` is 0.99, which reaches 63% of the specified learning rate in 100 steps and 95% in 300 steps. Steps where a Hessian-vector product is not done do not count toward the lr warmup.
