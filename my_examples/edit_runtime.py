def simulate_hnn(model, x0, steps, dt):
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = torch.as_tensor(x0, dtype=torch.float32).detach()
    trajectory = [x.tolist()]
    
    for _ in range(int(steps)):
        x = torch.as_tensor(model(x), dtype=torch.float32).detach()
        trajectory.append(x.tolist())
    
    traj = np.array(trajectory)
    q = traj[:, 0]
    p = traj[:, 1]
    t = np.arange(len(traj)) * float(dt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HNN Trajectory", fontsize=14)

    axes[0].plot(q, p, 'b-', linewidth=1.0)
    axes[0].set_xlabel('q (position)')
    axes[0].set_ylabel('p (momentum)')
    axes[0].set_title('Phase Portrait')
    axes[0].grid(True)
    axes[0].set_aspect('equal')

    axes[1].plot(t, q, 'b-', label='q (position)', linewidth=1.0)
    axes[1].plot(t, p, 'r-', label='p (momentum)', linewidth=1.0)
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('value')
    axes[1].set_title('Time Evolution')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('hnn_simulation.png', dpi=150)
    plt.show()
    print("Saved to hnn_simulation.png")


def simulate_heat_equation(solver, x0, steps, dt, snapshots=[0, 5, 10, 50]):
    import matplotlib.pyplot as plt
    import numpy as np

    x = torch.as_tensor(x0, dtype=torch.float32).detach()
    trajectory = [x.tolist()]

    for _ in range(int(steps)):
        x = torch.as_tensor(solver(x), dtype=torch.float32).detach()
        trajectory.append(x.tolist())

    traj = np.array(trajectory)
    x_grid = np.arange(traj.shape[1]) * 0.25

    plt.figure(figsize=(7, 4))
    for t in snapshots:
        idx = min(t, len(traj)-1)
        plt.plot(x_grid, traj[idx],  linewidth=2, label=f"t={t}")
    plt.xlabel("x")
    plt.ylabel("temperature")
    plt.title("physika plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return traj