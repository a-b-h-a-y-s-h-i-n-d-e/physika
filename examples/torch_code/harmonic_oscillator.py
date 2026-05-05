import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad
from physika.runtime import animate

# === Functions ===
def factorial(n):
    result = 1.0
    for i in range(int(1), int((n + 1))):
        result = (result * i)
    return result

def physika_cos(x):
    n = 10
    result = 0.0
    for i in range(int(0), int(n)):
        sign = ((-1) ** i)
        power = (x ** (2 * i))
        fact = factorial((2 * i))
        result = (result + ((sign * power) / fact))
    return result

def physika_sin(x):
    n = 10
    result = 0.0
    for i in range(int(0), int(n)):
        sign = ((-1) ** i)
        power = (x ** ((2 * i) + 1))
        fact = factorial(((2 * i) + 1))
        result = (result + ((sign * power) / fact))
    return result

def U(k, m, t, x0, v0):
    omega = ((k / m) ** 0.5)
    A = x0
    B = (v0 / omega)
    return ((A * physika_cos((omega * t))) + (B * physika_sin((omega * t))))

# === Program ===
k = 1.0
m = 1.0
x0 = 1.0
v0 = 0.0
physika_print(U(k, m, 0.0, x0, v0))
physika_print(U(k, m, 1.5708, x0, v0))
physika_print(U(k, m, 3.1416, x0, v0))
physika_print(U(k, m, 4.7124, x0, v0))
physika_print(U(k, m, 6.2832, x0, v0))
physika_print(U(k, m, 0.0, 0.0, 1.0))
physika_print(U(k, m, 1.5708, 0.0, 1.0))
physika_print(U(k, m, 3.1416, 0.0, 1.0))
animate(U, k, m, x0, v0, 0.0, 31.1416)
t0 = torch.tensor(0.0, requires_grad=True)
physika_print(compute_grad(lambda _dt0: real(U(k, m, t0, x0, v0)), t0))
t1 = torch.tensor(1.5708, requires_grad=True)
physika_print(compute_grad(lambda _dt1: real(U(k, m, t1, x0, v0)), t1))
t2 = torch.tensor(3.1416, requires_grad=True)
physika_print(compute_grad(lambda _dt2: real(U(k, m, t2, x0, v0)), t2))