"""ESGD-M (ESGD from "Equilibrated adaptive learning rates for non-convex optimization"
with quasi-hyperbolic momentum from "Quasi-hyperbolic momentum and Adam for deep
learning".
"""

import math

import torch
from torch import optim


class ESGD(optim.Optimizer):
    """ESGD-M (ESGD from "Equilibrated adaptive learning rates for non-convex optimization"
    with quasi-hyperbolic momentum from "Quasi-hyperbolic momentum and Adam for deep
    learning".

    To use this optimizer you must call .backward() with the create_graph=True option.
    Gradient accumulation steps and distributed training are currently not supported.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and squared Hessian diagonal estimate
            (default: (0.9, 0.999))
        nu (float, optional): the quasi-hyperbolic momentum 'nu' coefficient.
            If not specified, defaults to beta_1 (Nesterov momentum).
        lr_warmup (float, optional): exponential learning rate warmup coefficient
            (same units as betas, 0 means no warmup, closer to 1 means a longer
            warmup) (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        update_d_every (int, optional): update the squared Hessian diagonal
            estimate every update_d_every steps (default: 10)
        d_warmup (int, optional): update the squared Hessian diagonal estimate
            for the first d_warmup steps regardless of update_d_every (default: 20)

    .. _Equilibrated adaptive learning rates for non-convex optimization:
        https://proceedings.neurips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf

    .. _Quasi-hyperbolic momentum and Adam for deep learning:
        https://arxiv.org/abs/1810.06801
    """

    def __init__(self, params, lr=1, betas=(0.9, 0.999), nu=None, lr_warmup=0.99, eps=1e-4,
                 weight_decay=0., update_d_every=10, d_warmup=20):
        if not 0. <= lr:
            raise ValueError(f'Invalid learning rate: {lr:g}')
        if not 0. <= betas[0] < 1.:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]:g}')
        if not 0. <= betas[1] < 1.:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]:g}')
        if nu is not None and not 0. <= nu <= 1.:
            raise ValueError(f'Invalid nu parameter: {nu:g}')
        if not 0. <= lr_warmup < 1.:
            raise ValueError(f'Invalid lr warmup parameter: {lr_warmup:g}')
        if not 0. <= eps:
            raise ValueError(f'Invalid epsilon value: {eps:g}')
        if not 0. <= weight_decay:
            raise ValueError('Invalid weight_decay value: {weight_decay:g}')
        if not int(update_d_every) or not 1 <= update_d_every:
            raise ValueError(f'Invalid update_d_every parameter: {update_d_every}')
        if not int(d_warmup) or not 1 <= d_warmup:
            raise ValueError(f'Invalid d_warmup parameter: {d_warmup}')
        nu = betas[0] if nu is None else nu
        defaults = dict(lr=lr, betas=betas, nu=nu, lr_warmup=lr_warmup, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_d_every = update_d_every
        self.d_warmup = d_warmup
        self.steps = 0
        self.steps_since_d = 0

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.
        It contains three entries:
        * global_state - a dict holding global state.
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        global_state = {'update_d_every': self.update_d_every,
                        'd_warmup': self.d_warmup,
                        'steps': self.steps,
                        'steps_since_d': self.steps_since_d}
        return {'global_state': global_state, **super().state_dict()}

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Arguments:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self.update_d_every = state_dict['global_state']['update_d_every']
        self.d_warmup = state_dict['global_state']['d_warmup']
        self.steps = state_dict['global_state']['steps']
        self.steps_since_d = state_dict['global_state']['steps_since_d']

    def should_create_graph(self):
        """Returns True if the optimizer will update the squared Hessian diagonal estimate
        on the next call to .step() and thus you need to enable create_graph:

        >>> loss.backward(create_graph=optimizer.should_create_graph())
        """
        return self.steps < self.d_warmup or self.steps_since_d >= self.update_d_every

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute the squared Hessian diagonal estimate
        hvps_iter = None
        if self.should_create_graph():
            total = torch.tensor(0.)
            params, grads, vs = [], [], []
            with torch.enable_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError('ESGD does not support sparse gradients')
                        if p.grad.grad_fn is None:
                            msg = f'Gradient tensor shaped like {tuple(p.grad.shape)} does not have ' \
                                'a grad_fn. When calling loss.backward(), make sure the option ' \
                                'create_graph is set to True.'
                            raise RuntimeError(msg)
                        params.append(p)
                        grads.append(p.grad)
                        # Draw v from Rademacher distribution instead of normal as in ESGD paper
                        # to reduce variance.
                        vs.append(torch.randint_like(p.grad, 2) * 2 - 1)
                hvps = torch.autograd.grad(grads, params, grad_outputs=vs)
            hvps_iter = iter(hvps)
            self.steps_since_d = 0

        # Do the step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg_bias_corr'] = 1.
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared Hessian diagonal estimates
                    state['exp_avg_d_bias_corr'] = 1.
                    state['exp_avg_d'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Learning rate warmup cumulative product
                    state['lr_warmup_cumprod'] = 1.

                exp_avg, exp_avg_d = state['exp_avg'], state['exp_avg_d']
                beta1, beta2 = group['betas']
                nu = group['nu']

                # Update the running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_bias_corr'] *= beta1
                if hvps_iter is not None:
                    hvp = next(hvps_iter)
                    exp_avg_d.mul_(beta2).addcmul_(hvp, hvp.conj(), value=1 - beta2)
                    state['exp_avg_d_bias_corr'] *= beta2
                denom = (exp_avg_d.sqrt() / math.sqrt(1 - state['exp_avg_d_bias_corr'])).add_(group['eps'])

                # Learning rate schedule
                state['lr_warmup_cumprod'] *= group['lr_warmup']
                step_size = group['lr'] * (1 - state['lr_warmup_cumprod'])

                # Quasi-hyperbolic momentum
                exp_avg_est = torch.lerp(exp_avg, grad, 1 - nu)
                qhm_bias_corr = 1 - state['exp_avg_bias_corr'] * nu

                # Weight decay
                p.mul_(1 - group['weight_decay'] * step_size)

                # Do the step
                p.addcdiv_(exp_avg_est, denom, value=-step_size / qhm_bias_corr)

        self.steps += 1
        self.steps_since_d += 1

        return loss
