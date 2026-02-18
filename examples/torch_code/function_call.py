import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Functions ===
def f(x):
    return x

# === Program ===
physika_print(f(1.0))