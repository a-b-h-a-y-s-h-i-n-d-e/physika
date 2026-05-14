import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def magnitude(z):
    return torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))

# === Program ===
x = torch.tensor((3 + 1j), dtype=torch.complex64)
y = torch.tensor((5 + 3j), dtype=torch.complex64)
physika_print(x)
physika_print(y)
physika_print((x + y))
physika_print((x - y))
physika_print((x * y))
physika_print(magnitude(x))