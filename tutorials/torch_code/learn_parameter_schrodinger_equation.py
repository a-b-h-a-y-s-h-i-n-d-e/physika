import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i))]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    dx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * dx))
    return x

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_complex_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 1j) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i))]])
    return results

def append_row(x, row):
    rows = get_2d_array_num_rows(x)
    cols = get_1d_array_length(x[int(0)])
    new_rows = (rows + 1)
    new_array = zero_complex_2d_array(new_rows, cols)
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            new_array[int(i), int(j)] = x[int(i), int(j)]
    for j in range(int(0), int(cols)):
        new_array[int(rows), int(j)] = row[int(j)]
    return new_array

def schrodinger_rhs(psi, V, dx, hbar, mass):
    psi_xx = (((torch.roll(psi, (-1)) - (2 * psi)) + torch.roll(psi, 1)) / (dx ** 2))
    H_psi = (((-((hbar ** 2) / (2 * mass))) * psi_xx) + (V * psi))
    result = (((-1j) / hbar) * H_psi)
    return result

def make_potential(V_value):
    V = zero_1d_array(Nx)
    x = linspace((-200), 200, Nx)
    for i in range(int(0), int(Nx)):
        if torch.abs(x[int(i)] if isinstance(x[int(i)], torch.Tensor) else torch.tensor(float(x[int(i)]))) < 15:
            V[int(i)] = V_value
    return V

def RK4_step(psi, dt, V, dx, hbar, mass):
    k1 = schrodinger_rhs(psi, V, dx, hbar, mass)
    k2 = schrodinger_rhs((psi + ((0.5 * dt) * k1)), V, dx, hbar, mass)
    k3 = schrodinger_rhs((psi + ((0.5 * dt) * k2)), V, dx, hbar, mass)
    k4 = schrodinger_rhs((psi + (dt * k3)), V, dx, hbar, mass)
    psi_next = (psi + ((dt / 6.0) * (((k1 + (2 * k2)) + (2 * k3)) + k4)))
    return psi_next

def solver(V):
    x = linspace((-200), 200, Nx)
    psi0 = (((((1 / sigma) * torch.sqrt(3.14 if isinstance(3.14, torch.Tensor) else torch.tensor(float(3.14)))) ** 0.5) * torch.exp(((1j * k0) * x) if isinstance(((1j * k0) * x), torch.Tensor) else torch.tensor(float(((1j * k0) * x))))) * torch.exp(((-((x - x0) ** 2)) / (2 * (sigma ** 2))) if isinstance(((-((x - x0) ** 2)) / (2 * (sigma ** 2))), torch.Tensor) else torch.tensor(float(((-((x - x0) ** 2)) / (2 * (sigma ** 2)))))))
    history = torch.stack([torch.as_tensor(psi0).float()])
    counter = 0
    psi = psi0
    for i in range(int(0), int(Nt)):
        psi = RK4_step(psi, dt, V, dx, hbar, mass)
        counter = (counter + 1)
        if counter == 5:
            history = append_row(history, psi)
            counter = 0
    return history

def calculate_loss(barrier_height):
    V_current = make_potential(barrier_height)
    pred = solver(V_current)
    loss = torch.mean((torch.abs((pred - true_values) if isinstance((pred - true_values), torch.Tensor) else torch.tensor(float((pred - true_values)))) ** 2) if isinstance((torch.abs((pred - true_values) if isinstance((pred - true_values), torch.Tensor) else torch.tensor(float((pred - true_values)))) ** 2), torch.Tensor) else torch.tensor(float((torch.abs((pred - true_values) if isinstance((pred - true_values), torch.Tensor) else torch.tensor(float((pred - true_values)))) ** 2))))
    return loss

def adam(bh, g, m, v, t, lr):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    m_new = ((beta1 * m) + ((1.0 - beta1) * g))
    v_new = ((beta2 * v) + ((1.0 - beta2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (beta1 ** t)))
    v_hat = (v_new / (1.0 - (beta2 ** t)))
    bh_new = (bh - ((lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + eps)))
    return torch.stack([torch.as_tensor(bh_new).float(), torch.as_tensor(m_new).float(), torch.as_tensor(v_new).float(), torch.as_tensor((t + 1.0)).float()])

# === Program ===
Nx = 102
Nt = 327
x = linspace((-200), 200, Nx)
dx = 0.391
hbar = 1.0
mass = 1.0
cfl_factor = 0.2
dt = ((cfl_factor * ((mass * dx) ** 2)) / hbar)
t_final = 100.0
x0 = (-50.0)
k0 = 2.0
sigma = 10.0
psi0 = (((((1 / sigma) * torch.sqrt(3.14 if isinstance(3.14, torch.Tensor) else torch.tensor(float(3.14)))) ** 0.5) * torch.exp(((1j * k0) * x) if isinstance(((1j * k0) * x), torch.Tensor) else torch.tensor(float(((1j * k0) * x))))) * torch.exp(((-((x - x0) ** 2)) / ((2 * sigma) ** 2)) if isinstance(((-((x - x0) ** 2)) / ((2 * sigma) ** 2)), torch.Tensor) else torch.tensor(float(((-((x - x0) ** 2)) / ((2 * sigma) ** 2))))))
V = make_potential(1.8)
true_values = solver(V)
guess_barrier_height = torch.tensor(6.0, requires_grad=True)
guess_V = make_potential(guess_barrier_height)
guess_values = solver(guess_V)
m_adam = 0.0
v_adam = 0.0
t_adam = 1.0
lr = 0.1
epochs = 1
for i in range(int(0), int(epochs)):
    physika_print(i)
    g = compute_grad(calculate_loss, guess_barrier_height)
    result = adam(guess_barrier_height, g, m_adam, v_adam, t_adam, lr)
    guess_barrier_height = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
pred_V = make_potential(guess_barrier_height)
pred_results = solver(pred_V)