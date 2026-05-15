import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# spatial grid
Nx = 1024
x = np.linspace(-200.0, 200.0, Nx)
dx = x[1] - x[0]


# physical constants
hbar = 1.0
m = 1.0

# CFL
cfl_factor = 0.2
dt = cfl_factor * (m * dx**2) / hbar


# total number of timesteps
t_final = 100.0
Nt = int(np.ceil(t_final / dt))


# reference timesteps
Nt_ref = 10000
steps_ref = 10  # steps per frame

# scale steps per frame for animation speed
steps_per_frame = int(np.ceil(steps_ref * (Nt / Nt_ref)))


# initial wave function (gaussian wave packet)

x0 = -50.0
k0 = 2.0
sigma = 10.0

psi0 = (
    (1 / (sigma * np.sqrt(np.pi))) ** 0.5 
    * np.exp(1j * k0 * x)
    * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
)



# potential barrier
V = np.zeros(Nx)

# barrier centered at x = 0, set potential to V0 within [-a, a]
#V[np.abs(x) < 15.0] = 1.8
V[np.abs(x) < 15.0] = 1.8


# central finite-difference 
def schrodinger_rhs(psi, V, dx, hbar, m):
    psi_xx = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dx**2
    
    # hamiltonian operator acting on psi
    H_psi = -(hbar**2 / (2*m)) * psi_xx + V * psi
    
    # return with complex value
    return -1j / hbar * H_psi


# RK4 step

def RK4_step(psi, dt, V, dx, hbar, m):
    k1 = schrodinger_rhs(psi, V, dx, hbar, m)
    k2 = schrodinger_rhs(psi + 0.5 * dt * k1, V, dx, hbar, m)
    k3 = schrodinger_rhs(psi + 0.5 * dt * k2, V, dx, hbar, m)
    k4 = schrodinger_rhs(psi + dt * k3, V, dx, hbar, m)
    psi_next = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi_next

def solver(V):
    psi0 = (
        (1 / (sigma * np.sqrt(np.pi))) ** 0.5 
        * np.exp(1j * k0 * x)
        * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    )
    psi = psi0.copy()
    history = []
    save_every = 5
    for i in range(Nt):
        psi = RK4_step(psi, dt, V, dx, hbar, m)
        if i % save_every == 0:
            history.append(psi.copy())
    return history


def update(frame, history, line_prob, line_re, line_im):
    """
    Evolves the wave function for several time steps per frame to smooth the animation,
    and updates the probability density and real/imaginary parts of the wave function.
    """
    psi = history[frame]
    line_prob.set_ydata(np.abs(psi) ** 2)
    line_re.set_ydata(np.real(psi))
    line_im.set_ydata(np.imag(psi))
    return line_prob, line_re, line_im


def initialize_plot():
    """
    Sets up the figure and initial plots for the animation.
    """
    fig, (ax_prob, ax_reim) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    V_scale_prob = np.max(np.abs(psi0)) ** 2 * 1.5
    V_scale_reim = np.max(np.abs(psi0)) * 1.5

    ax_prob.plot(x, V / 1.8 * V_scale_prob, "k--", lw=1.5, label="Potential")
    (line_prob,) = ax_prob.plot(
        x, np.abs(psi0) ** 2, "b-", lw=2, label=r"$|\psi(x,t)|^2$"
    )
    ax_prob.set_ylabel(r"$|\psi(x,t)|^2$")
    ax_prob.set_title("Quantum Tunneling")
    ax_prob.legend(loc="upper right")

    ax_reim.plot(x, V / 1.8 * V_scale_reim, "k--", lw=1.5, label="Potential")
    (line_re,) = ax_reim.plot(x, np.real(psi0), "b-", lw=2, label=r"Re{$\psi$}")
    (line_im,) = ax_reim.plot(x, np.imag(psi0), "r-", lw=2, label=r"Im{$\psi$}")

    return fig, line_prob, line_re, line_im


def create_plot(history):
    fig, line_prob, line_re, line_im = initialize_plot()
    ani = animation.FuncAnimation(
        fig,
        update,
        fargs=(history, line_prob, line_re, line_im),
        frames= len(history),
        interval=30,
        blit=False,
    )

    plt.tight_layout()
    plt.show()    



history = solver(V)

#create_plot(history)

true_values = solver(V)
true_values = np.array(true_values)
#create_plot(history)

def calculate_loss(V):
    predictions = np.array(solver(V))
    loss = np.mean(np.abs(predictions - true_values)**2)
    return loss


def make_potential(V0):
    V = np.zeros(Nx)
    V[np.abs(x) < 15.0] = V0
    return V

new_V = make_potential(1.8)
print(calculate_loss(new_V))
