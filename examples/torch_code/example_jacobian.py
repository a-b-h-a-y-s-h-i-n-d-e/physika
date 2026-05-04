import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f_vec(state):
    x = state[int(0)]
    y = state[int(1)]
    fx = ((x ** 2) + y)
    fy = (x + (y ** 2))
    return torch.stack([torch.as_tensor(fx).float(), torch.as_tensor(fy).float()])

def lotka_volterra(state, θ):
    x = state[int(0)]
    y = state[int(1)]
    α = θ[int(0)]
    β = θ[int(1)]
    γ = θ[int(2)]
    δ = θ[int(3)]
    dx = ((α * x) - ((β * x) * y))
    dy = (((δ * x) * y) - (γ * y))
    return torch.stack([torch.as_tensor(dx).float(), torch.as_tensor(dy).float()])

# === Program ===
state = torch.as_tensor(torch.tensor([1.0, 2.0])).float().requires_grad_(True)
J = compute_grad(f_vec, state)
physika_print(J)
state = torch.as_tensor(torch.tensor([10.0, 1.0])).float().requires_grad_(True)
θ = torch.tensor([1.5, 1.0, 3.0, 1.0])
J_state = compute_grad(lotka_volterra, state, θ, 0)
physika_print(J_state)
J_theta = compute_grad(lotka_volterra, state, θ, 1)
physika_print(J_theta)