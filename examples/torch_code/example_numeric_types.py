import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
a = int(10)
b = int(3)
z_add = (a + b)
z_array = torch.tensor([1, 2])
physika_print(a)
physika_print(b)
physika_print(z_add)
physika_print(z_array)
x = 3.14
y = 2.0
r_mul = (x * y)
physika_print(x)
physika_print(y)
physika_print(r_mul)
z_number = int(1)
r_number = 2.0
results = (z_number + r_number)
physika_print(results)
neg_int = int((-7))
neg_float = (-3.14)
physika_print(neg_int)
physika_print(neg_float)