import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

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

def wave_equation(u, Δx, Δy, c):
    lap = zero_2d_array(nx, ny)
    lap[1:(nx - 1), 1:(ny - 1)] = ((c ** 2) * ((((u[0:(nx - 2), 1:(ny - 1)] - (2 * u[1:(nx - 1), 1:(ny - 1)])) + u[2:nx, 1:(ny - 1)]) / (Δx ** 2)) + (((u[1:(nx - 1), 0:(ny - 2)] - (2 * u[1:(nx - 1), 1:(ny - 1)])) + u[1:(nx - 1), 2:ny]) / (Δy ** 2))))
    return lap

def solver(c, u0, v0, Δx, Δy, Δt, nt):
    u_prev = u0
    u_curr = u0
    for step in range(int(0), int(nt)):
        accel = wave_equation(u_curr, Δx, Δy, c)
        u_next = (((2 * u_curr) - u_prev) + ((Δt ** 2) * accel))
        u_next[:, int(0)] = 0
        u_next[:, int((ny - 1))] = 0
        u_next[int(0), :] = 0
        u_next[int((nx - 1)), :] = 0
        u_prev = u_curr
        u_curr = u_next
    return u_curr

def calculate_loss(c):
    predictions = solver(c, u0, v0, Δx, Δy, Δt, nt)
    diff = (predictions - true_solution)
    loss = torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))
    return loss

def adam(c, g, m, v, t, lr):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    m_new = ((beta1 * m) + ((1.0 - beta1) * g))
    v_new = ((beta2 * v) + ((1.0 - beta2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (beta1 ** t)))
    v_hat = (v_new / (1.0 - (beta2 ** t)))
    c_new = (c - ((lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + eps)))
    return torch.stack([torch.as_tensor(c_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
Lx = 1.0
Ly = 1.0
nx = 40
ny = 40
tf = 2.0
Δx = (Lx / (nx - 1))
Δy = (Ly / (ny - 1))
true_c = 1.0
cfl = 0.4
Δt = (cfl / (5.0 * torch.sqrt((((1 / Δx) ** 2) + ((1 / Δy) ** 2)) if isinstance((((1 / Δx) ** 2) + ((1 / Δy) ** 2)), torch.Tensor) else torch.tensor(float((((1 / Δx) ** 2) + ((1 / Δy) ** 2)))))))
nt = 50
x = linspace(0, Lx, nx)
y = linspace(0, Ly, ny)
pi = 3.14
u0 = zero_2d_array(nx, ny)
for i in range(int(0), int(nx)):
    for j in range(int(0), int(ny)):
        u0[int(i), int(j)] = (torch.sin(((2 * pi) * x[int(i)]) if isinstance(((2 * pi) * x[int(i)]), torch.Tensor) else torch.tensor(float(((2 * pi) * x[int(i)])))) * torch.sin((pi * y[int(j)]) if isinstance((pi * y[int(j)]), torch.Tensor) else torch.tensor(float((pi * y[int(j)])))))
v0 = zero_2d_array(nx, ny)
true_solution = solver(true_c, u0, v0, Δx, Δy, Δt, nt)
c = torch.tensor(3.0, requires_grad=True)
m_adam = 0.0
v_adam = 0.0
t_adam = 1.0
lr = 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    physika_print(i)
    g = compute_grad(calculate_loss, c)
    result = adam(c, g, m_adam, v_adam, t_adam, lr)
    c = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    physika_print(c)
pred_solution = solver(c, u0, v0, Δx, Δy, Δt, nt)