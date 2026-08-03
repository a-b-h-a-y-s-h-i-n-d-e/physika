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

def linspace(start, end, n):
    x = zero_1d_array(n)
    Δx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * Δx))
    return x

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def central_difference_x(f):
    diff = zero_2d_array(n_points, n_points)
    diff[1:(n_points - 1), 1:(n_points - 1)] = ((f[1:(n_points - 1), 2:n_points] - f[1:(n_points - 1), 0:(n_points - 2)]) / (2 * element_length))
    return diff

def central_difference_y(f):
    diff = zero_2d_array(n_points, n_points)
    diff[1:(n_points - 1), 1:(n_points - 1)] = ((f[2:n_points, 1:(n_points - 1)] - f[0:(n_points - 2), 1:(n_points - 1)]) / (2 * element_length))
    return diff

def laplace(f):
    diff = zero_2d_array(n_points, n_points)
    diff[1:(n_points - 1), 1:(n_points - 1)] = (((((f[1:(n_points - 1), 0:(n_points - 2)] + f[0:(n_points - 2), 1:(n_points - 1)]) + f[1:(n_points - 1), 2:n_points]) + f[2:n_points, 1:(n_points - 1)]) - (4 * f[1:(n_points - 1), 1:(n_points - 1)])) / (element_length ** 2))
    return diff

def solver(ρ):
    u_prev = zero_2d_array(n_points, n_points)
    v_prev = zero_2d_array(n_points, n_points)
    p_prev = zero_2d_array(n_points, n_points)
    for i in range(int(0), int(n_iterations)):
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)
        u_tent = (u_prev + (time_step_length * ((-((u_prev * d_u_prev__d_x) + (v_prev * d_u_prev__d_y))) + (ν * laplace__u_prev))))
        v_tent = (v_prev + (time_step_length * ((-((u_prev * d_v_prev__d_x) + (v_prev * d_v_prev__d_y))) + (ν * laplace__v_prev))))
        u_tent[int(0), :] = 0.0
        u_tent[int((-1)), :] = horizontal_velocity_top
        u_tent[:, int(0)] = 0.0
        u_tent[:, int((-1))] = 0.0
        v_tent[int(0), :] = 0.0
        v_tent[int((-1)), :] = 0.0
        v_tent[:, int(0)] = 0.0
        v_tent[:, int((-1))] = 0.0
        d_u_tent__d_x = central_difference_x(u_tent)
        d_v_tent__d_y = central_difference_y(v_tent)
        rhs = ((ρ / time_step_length) * (d_u_tent__d_x + d_v_tent__d_y))
        for k in range(int(0), int(n_pressure_poisson_iterations)):
            p_next = zero_2d_array(n_points, n_points)
            p_next[1:(-1), 1:(-1)] = (0.25 * ((((p_prev[1:(-1), :(-2)] + p_prev[:(-2), 1:(-1)]) + p_prev[1:(-1), 2:]) + p_prev[2:, 1:(-1)]) - ((element_length ** 2) * rhs[1:(-1), 1:(-1)])))
            p_next[:, int((-1))] = p_next[:, int((-2))]
            p_next[int(0), :] = p_next[int(1), :]
            p_next[:, int(0)] = p_next[:, int(1)]
            p_next[int((-1)), :] = 0.0
            p_prev = p_next
        d_p_next__d_x = central_difference_x(p_next)
        d_p_next__d_y = central_difference_y(p_next)
        u_next = (u_tent - ((time_step_length / ρ) * d_p_next__d_x))
        v_next = (v_tent - ((time_step_length / ρ) * d_p_next__d_y))
        u_next[int(0), :] = 0.0
        u_next[:, int(0)] = 0.0
        u_next[:, int((-1))] = 0.0
        u_next[int((-1)), :] = horizontal_velocity_top
        v_next[int(0), :] = 0.0
        v_next[:, int(0)] = 0.0
        v_next[:, int((-1))] = 0.0
        v_next[int((-1)), :] = 0.0
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
    return torch.stack([torch.as_tensor(u_prev), torch.as_tensor(v_prev), torch.as_tensor(p_prev)])

def calculate_loss(ρ):
    predictions = solver(ρ)
    pred_u = predictions[int(0)]
    pred_v = predictions[int(1)]
    pred_p = predictions[int(2)]
    loss_u = torch.mean(((pred_u - true_u) ** 2) if isinstance(((pred_u - true_u) ** 2), torch.Tensor) else torch.tensor(float(((pred_u - true_u) ** 2))))
    loss_v = torch.mean(((pred_v - true_v) ** 2) if isinstance(((pred_v - true_v) ** 2), torch.Tensor) else torch.tensor(float(((pred_v - true_v) ** 2))))
    loss_p = torch.mean(((pred_p - true_p) ** 2) if isinstance(((pred_p - true_p) ** 2), torch.Tensor) else torch.tensor(float(((pred_p - true_p) ** 2))))
    loss = ((loss_u + loss_v) + loss_p)
    return loss

def adam(ρ, g, m, v, t, lr):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    m_new = ((beta1 * m) + ((1.0 - beta1) * g))
    v_new = ((beta2 * v) + ((1.0 - beta2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (beta1 ** t)))
    v_hat = (v_new / (1.0 - (beta2 ** t)))
    ρ_new = (ρ - ((lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + eps)))
    return torch.stack([torch.as_tensor(ρ_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
n_points = 21
domain_size = 1.0
n_iterations = 500
time_step_length = 0.001
ν = 0.1
true_ρ = 1.0
horizontal_velocity_top = 1.0
n_pressure_poisson_iterations = 10
stability_safety_factor = 0.5
element_length = (domain_size / (n_points - 1))
x = linspace(0.0, domain_size, n_points)
y = linspace(0.0, domain_size, n_points)
f = zero_2d_array(n_points, n_points)
for i in range(int(0), int(n_points)):
    for j in range(int(0), int(n_points)):
        t_x = (j * element_length)
        t_y = (i * element_length)
        f[int(i), int(j)] = ((t_x ** 2) + (t_y ** 2))
n_iterations = 5
true_solution = solver(true_ρ)
true_u = true_solution[int(0)]
true_v = true_solution[int(1)]
true_p = true_solution[int(2)]
ρ = torch.tensor(3.0, requires_grad=True)
m_adam = 0.0
v_adam = 0.0
t_adam = 1.0
lr = 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    physika_print(i)
    g = compute_grad(calculate_loss, ρ)
    result = adam(ρ, g, m_adam, v_adam, t_adam, lr)
    ρ = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    physika_print(ρ)
physika_print(ρ)
pred_solution = solver(ρ)
pred_u = pred_solution[int(0)]
pred_v = pred_solution[int(1)]
pred_p = pred_solution[int(2)]
X = zero_2d_array(n_points, n_points)
Y = zero_2d_array(n_points, n_points)
for i in range(int(0), int(n_points)):
    for j in range(int(0), int(n_points)):
        X[int(i), int(j)] = (j * element_length)
        Y[int(i), int(j)] = (i * element_length)