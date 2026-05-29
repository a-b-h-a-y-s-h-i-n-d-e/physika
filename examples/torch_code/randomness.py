import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Program ===
m_true = 3.0
b_true = 1.0
m_hat = 2.0
b_hat = 0.0
μ = torch.tensor(2.0, requires_grad=True)
σ = torch.tensor(0.5, requires_grad=True)
n = int(100)
x = torch.distributions.Normal(μ, σ).rsample((int(n),))
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
physika_print(loss)
grad_mu = compute_grad(loss, μ)
grad_sigma = compute_grad(loss, σ)
physika_print(grad_mu)
physika_print(grad_sigma)
lr = 0.01
for step in range(int(0), int(300)):
    x = torch.distributions.Normal(μ, σ).rsample((int(n),))
    y_true = ((m_true * x) + b_true)
    y_pred = ((m_hat * x) + b_hat)
    loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
    grad_mu = compute_grad(loss, μ)
    grad_sigma = compute_grad(loss, σ)
    μ = (μ - (lr * grad_mu))
    σ = (σ - (lr * grad_sigma))
physika_print(μ)
physika_print(σ)
x = torch.distributions.Normal(μ, σ).rsample((int(n),))
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
physika_print(loss)
z = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)) for _fi_i in range(int(10)) for i in [torch.tensor(float(_fi_i))]])
physika_print(z)
y = torch.distributions.Normal(μ, σ).rsample((int(10), int(10),))
x = 3
physika_print(sample_normal(x))