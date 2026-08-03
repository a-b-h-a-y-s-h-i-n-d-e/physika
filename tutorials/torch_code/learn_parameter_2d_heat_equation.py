import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    Δx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * Δx))
    return x

def heat_equation(T, Δx, Δy, α):
    f = zero_2d_array(nx, ny)
    f[1:(nx - 1), 1:(ny - 1)] = (α * ((((T[0:(nx - 2), 1:(ny - 1)] - (2 * T[1:(nx - 1), 1:(ny - 1)])) + T[2:nx, 1:(ny - 1)]) / (Δx ** 2)) + (((T[1:(nx - 1), 0:(ny - 2)] - (2 * T[1:(nx - 1), 1:(ny - 1)])) + T[1:(nx - 1), 2:ny]) / (Δy ** 2))))
    return f

def solver(α, T0, Δx, Δy, Δt, nt):
    T = T0
    for step in range(int(0), int(nt)):
        T = (T + (Δt * heat_equation(T, Δx, Δy, α)))
        T[:, int(0)] = 0
        T[:, int((ny - 1))] = 0
        T[int(0), :] = 0
        T[int((nx - 1)), :] = 0
    return T

def calculate_loss(α):
    predictions = solver(α, T0, Δx, Δy, Δt, nt)
    diff = (predictions - true_solution)
    loss = torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))
    return loss

def adam(α, g, m, v, t, lr):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    m_new = ((beta1 * m) + ((1.0 - beta1) * g))
    v_new = ((beta2 * v) + ((1.0 - beta2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (beta1 ** t)))
    v_hat = (v_new / (1.0 - (beta2 ** t)))
    α_new = (α - ((lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + eps)))
    return torch.stack([torch.as_tensor(α_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
true_α = 2.0
Lx = 1.0
Ly = 1.0
nx = 40
ny = 40
tf = 10
Δx = (Lx / (nx - 1))
Δy = (Ly / (ny - 1))
fourier = 0.49
Δt = ((fourier / (((1 / Δx) ** 2) + ((1 / Δy) ** 2))) / 10.0)
nt = 100
T1, T2, T3, T4 = 0, 0, 0, 0
x = linspace(0, Lx, nx)
y = linspace(0, Ly, ny)
T0 = torch.stack([torch.stack([torch.exp(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2))) if isinstance(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2))), torch.Tensor) else torch.tensor(float(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2)))))) for _fi_j in range(int(ny)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
true_solution = solver(true_α, T0, Δx, Δy, Δt, nt)
α = torch.tensor(4.0, requires_grad=True)
guess_solution = solver(α, T0, Δx, Δy, Δt, nt)
m_adam = 0.0
v_adam = 0.0
t_adam = 1.0
lr = 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    physika_print(i)
    g = compute_grad(calculate_loss, α)
    result = adam(α, g, m_adam, v_adam, t_adam, lr)
    α = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    physika_print(α)
physika_print(α)
pred_solution = solver(α, T0, Δx, Δy, Δt, nt)