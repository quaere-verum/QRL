import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 / math.pi)
torch.manual_seed(0)

def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return (1.0 + torch.erf(x / SQRT_TWO)) / 2.0

def quantile_huber_loss(
    quantile_estimate: torch.Tensor,
    quantile_fractions: torch.Tensor,
    target: torch.Tensor,
    mse_bound: float,
) -> torch.Tensor:
    batch_size = target.shape[0]
    u = target[:, torch.newaxis, :] - quantile_estimate[:, :, torch.newaxis]
    u_abs = torch.abs(u)
    huber_loss = torch.where(
        u_abs < mse_bound,
        0.5 * torch.pow(u, 2),
        mse_bound * (u_abs - 0.5 * mse_bound)
    )
    multiplier = torch.abs(
        quantile_fractions[:, torch.newaxis]
        - (u.detach() < 0).float()
    ) / mse_bound
 
    return torch.sum(huber_loss * multiplier) / (batch_size * quantile_fractions.shape[0])

def generalised_quantile_huber_loss(
    quantile_estimate: torch.nn.Parameter,
    target: torch.Tensor,
    quantile_fractions: torch.Tensor,
) -> torch.Tensor:
    
    b = torch.abs(torch.std(quantile_estimate) - torch.std(target))

    u = target[torch.newaxis, :] - quantile_estimate[:, torch.newaxis]
    u_abs = torch.abs(u)
    multiplier = torch.abs(quantile_fractions[:, torch.newaxis] - (u < 0).float())

    generalised_huber_loss = (
        u_abs
        * (1.0 - 2.0 * standard_normal_cdf(-u_abs / b))
        + b * SQRT_TWO_PI * (torch.exp(-u ** 2 / (2 * b ** 2)) - 1.0)
    )
    return torch.sum(generalised_huber_loss * multiplier) / len(quantile_fractions)

def test_quantile_huber_loss():
    nr_quantiles = 100
    quantile_fractions = torch.linspace(1 / nr_quantiles, 1.0 - 1 / nr_quantiles, nr_quantiles)
    quantile_estimate = torch.nn.Parameter(torch.randn(len(quantile_fractions)), requires_grad=True)
    base_distribution = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0]))
    mse_bound = 0.0001
    optim = torch.optim.Adam([{"params": quantile_estimate}], lr=0.1)

    nr_steps = 200
    plot_interval = 50
    sample_size = 512
    cmap = plt.get_cmap("viridis", nr_steps // plot_interval + 1)
    plt.plot(quantile_fractions.numpy(), base_distribution.icdf(quantile_fractions).numpy())
    for step in tqdm(range(nr_steps)):
        samples = base_distribution.sample(sample_shape=(sample_size,)).broadcast_to((sample_size, nr_quantiles))
        loss = quantile_huber_loss(
            quantile_estimate=quantile_estimate.broadcast_to((sample_size, quantile_estimate.shape[0])), 
            target=samples, 
            quantile_fractions=quantile_fractions, 
            mse_bound=mse_bound
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % plot_interval == 0:
            plt.plot(quantile_fractions.numpy(), quantile_estimate.data.detach().numpy().copy(), c=cmap(step // plot_interval), alpha=0.5)
    plt.grid()
    plt.show()

    plt.plot(quantile_fractions.numpy(), base_distribution.icdf(quantile_fractions).numpy(), linestyle=":")
    plt.plot(quantile_fractions.numpy(), quantile_estimate.data.detach().numpy().copy(), alpha=0.5)
    plt.grid()
    plt.show()

def test_generalised_quantile_huber_loss():
    nr_quantiles = 100
    quantile_fractions = torch.linspace(1 / nr_quantiles, 1.0 - 1 / nr_quantiles, nr_quantiles)
    quantile_estimate = torch.nn.Parameter(torch.randn(len(quantile_fractions)), requires_grad=True)
    base_distribution = torch.distributions.LogNormal(torch.tensor([0.0]), torch.tensor([2.0]))
    optim = torch.optim.Adam([{"params": quantile_estimate}], lr=0.01)

    nr_steps = 4000
    plot_interval = 200
    sample_size = 512
    cmap = plt.get_cmap("viridis", nr_steps // plot_interval + 1)
    plt.plot(quantile_fractions.numpy(), base_distribution.icdf(quantile_fractions).numpy())
    for step in tqdm(range(nr_steps)):
        samples = base_distribution.sample(sample_shape=(sample_size,))
        target = torch.quantile(samples, quantile_fractions)
        loss = generalised_quantile_huber_loss(quantile_estimate=quantile_estimate, target=target, quantile_fractions=quantile_fractions)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % plot_interval == 0:
            plt.plot(quantile_fractions.numpy(), target.numpy(), c=cmap(step // plot_interval), linestyle=":")
    plt.grid()
    plt.show()

def main():
    test_quantile_huber_loss()
    # test_generalised_quantile_huber_loss()

if __name__ == "__main__":
    main()