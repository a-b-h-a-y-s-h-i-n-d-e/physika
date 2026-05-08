import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def H(J, h, spins):
    nn_products = torch.stack([(spins[int(i)] * spins[int((i + 1))]) for _fi_i in range(int((n - 1))) for i in [torch.tensor(float(_fi_i))]])
    nn_sum = torch.sum(nn_products)
    field_sum = torch.sum(spins)
    return (((-J) * nn_sum) - (h * field_sum))

# === Program ===
n = 16.0
J = torch.tensor(2.0, requires_grad=True)
h = torch.tensor(0.1, requires_grad=True)
β = 5.0
b = torch.distributions.Bernoulli(0.5).sample((int(n),)).detach()
spins = torch.stack([((2.0 * b[int(i)]) - 1.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]])
physika_print(spins)
M = (torch.sum(spins) / n)
physika_print(M)
physika_print(H(J, h, spins))
log_p = ((-β) * H(J, h, spins))
physika_print(log_p)
dH_dJ = compute_grad(lambda _dJ: H(_dJ, h, spins), J)
dH_dh = compute_grad(lambda _dh: H(J, _dh, spins), h)
physika_print(dH_dJ)
physika_print(dH_dh)