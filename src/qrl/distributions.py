"""Bounded Gaussian used by stochastic policies (same transform as rust-rl)."""
import math
import torch
from torch.distributions import Distribution, Independent, Normal, constraints, kl_divergence, register_kl
from torch.nn.functional import softplus


class SquashedGaussian(Distribution):
    """a = center + scale*tanh(z), z ~ diagonal Normal.

    Likelihoods include both Jacobians. Keep z for replay: rounded saturated
    actions cannot be inverted reliably. Entropy uses a reparameterized sample;
    KL is exact in latent space when both distributions share the transform.
    """
    arg_constraints = {}
    has_rsample = True

    def __init__(self, mean, std, low, high):
        self.base_dist = Independent(Normal(mean, std), 1)
        self.low = torch.as_tensor(low, dtype=mean.dtype, device=mean.device)
        self.high = torch.as_tensor(high, dtype=mean.dtype, device=mean.device)
        if not (torch.isfinite(self.low).all() and torch.isfinite(self.high).all()
                and (self.high > self.low).all() and torch.isfinite(self.high-self.low).all()):
            raise ValueError('Squashed Gaussian requires finite, strictly ordered action bounds')
        self.scale = (self.high-self.low)/2
        self.center = self.low+self.scale
        super().__init__(self.base_dist.batch_shape, self.base_dist.event_shape)

    @property
    def support(self):
        return constraints.independent(constraints.interval(self.low, self.high), 1)

    def squash(self, latent):
        # Guard one-ulp affine roundoff at finite-precision endpoints.
        return (self.center+self.scale*latent.tanh()).clamp(self.low, self.high)

    def _log_jacobian(self, latent):
        # Stable even when tanh(latent) has rounded to +/-1.
        return (self.scale.log()+2*(math.log(2.)-latent-softplus(-2*latent))).sum(-1)

    def sample_with_latent(self, sample_shape=torch.Size()):
        latent = self.base_dist.sample(sample_shape)
        return self.squash(latent), latent

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.sample_with_latent(sample_shape)[0]

    def rsample(self, sample_shape=torch.Size()):
        return self.squash(self.base_dist.rsample(sample_shape))

    def log_prob(self, value, *, latents=None):
        if self._validate_args:
            self._validate_sample(value)
        if latents is None:
            normalized = (value-self.center)/self.scale
            if (normalized.abs() >= 1).any():
                raise ValueError('Actions at rounded bounds require their stored Gaussian latents')
            latents = torch.atanh(normalized)
        else:
            if latents.shape != value.shape or not torch.isfinite(latents).all():
                raise ValueError('Invalid Gaussian latent shape or nonfinite latent')
            tolerance = 1e-5*self.scale.clamp(min=1.)
            if ((value-self.squash(latents)).abs() > tolerance).any():
                raise ValueError('Action does not match its stored Gaussian latent')
        return self.base_dist.log_prob(latents)-self._log_jacobian(latents)

    def entropy(self):
        return self.base_dist.entropy()+self._log_jacobian(self.base_dist.rsample())


@register_kl(SquashedGaussian, SquashedGaussian)
def _kl_squashed_gaussian(p, q):
    if not (torch.equal(p.low, q.low) and torch.equal(p.high, q.high)):
        raise ValueError('Gaussian KL requires matching action transforms')
    return kl_divergence(p.base_dist, q.base_dist)
