Parameter learning 2D PDEs
======================================


In this tutorial we will learn how to estimate/learn parameter of different 
2D PDEs such as Heat equation, Wave equation, Navier stokes equation etc. in Physika.
The tutorial is divided into 3 parts as per each PDE example.

We recommend first exploring our 1D PDE tutorials on the Heat and Wave equations, as 
they introduce the key concepts and techniques that are used throughout the 2D examples in this tutorial.


2D Heat equation
----------------

.. math::

   \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)

where :math:`u(x, y, t)` is the temperature field, :math:`x` and :math:`y` are
the spatial coordinates, :math:`t` is time, and :math:`\alpha` is the
thermal diffusivity, the parameter we want to learn.


Helper functions
^^^^^^^^^^^^^^^^^^^


.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x


Set Up the Domain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    true_α: ℝ = 2.0
    Lx: ℝ = 1.0
    Ly: ℝ = 1.0

    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 10

    dx: ℝ = Lx / (nx - 1)
    dy: ℝ = Ly / (ny - 1)

``Lx`` and ``Ly`` define the length on both the axis, ``nx`` and ``ny`` set the
number of grid points along each axis, and ``dx`` and ``dy`` are the
resulting grid spacings.


Time stepping
^^^^^^^^^^^^^

.. code-block::

    fourier: ℝ = 0.49
    dt: ℝ = fourier / (1/dx**2 + 1/dy**2) / 10.0
    nt: ℝ = 100

The time step ``dt`` is computed from the Fourier number to satisfy the
stability condition for explicit time-stepping, and ``nt`` is the total
number of time steps to simulate.


Grid and initial condition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    # Boundary conditions
    T1: ℝ = 0
    T2: ℝ = 0
    T3: ℝ = 0
    T4: ℝ = 0

    T: ℝ[nx, ny] = zero_2d_array(nx, ny)

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)


    T0: ℝ[nx, ny] = zero_2d_array(nx, ny)
    for i:ℕ(0, nx):
        for j:ℕ(0, ny):
            T0[i, j] = exp(-20 * (((x[i]-0.5)**2) + ((y[j]-0.5)**2)))

``T1``–``T4`` set the (zero) Dirichlet boundary conditions on the four edges
of the plate. We then build the spatial grid using ``x`` and ``y``, and
initialize the temperature field ``T0`` as a Gaussian pulse centered at the
middle of the domain — this is the initial condition the PDE will evolve
forward in time from.



Discretize the Heat equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using second-order central differences for the spatial derivatives, the 2D
heat equation becomes:

.. math::

    \begin{align*}
    f_{i,j} = \alpha \left(
    \frac{T_{i-1,j} - 2T_{i,j} + T_{i+1,j}}{\Delta x^2} +
    \frac{T_{i,j-1} - 2T_{i,j} + T_{i,j+1}}{\Delta y^2}
    \right)
    \end{align*}

.. code-block:: text

    def heat_equation(T: ℝ[m, n], dx: ℝ, dy: ℝ, α: ℝ): ℝ[m, n]:
        f: ℝ[m, n] = zero_2d_array(nx, ny)
        f[1:nx-1, 1:ny-1] = α * (
            ((T[0:nx-2, 1:ny-1] - 2 * T[1:nx-1, 1:ny-1] + T[2:nx, 1:ny-1]) / (dx**2)) +
            ((T[1:nx-1, 0:ny-2] - 2 * T[1:nx-1, 1:ny-1] + T[1:nx-1, 2:ny]) / (dy**2))
        )
        return f

Build the solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: text

    def solver(α: ℝ, T0: ℝ[m, n], dx: ℝ, dy: ℝ, dt: ℝ, nt: ℝ): ℝ[m, n]:
        T: ℝ[m, n] = T0
        for step:ℕ(0, nt):
            T = T + dt * heat_equation(T, dx, dy, α)
            T[:, 0] = 0
            T[:, ny-1] = 0
            T[0, :] = 0
            T[nx-1, :] = 0
        return T

    true_solution: ℝ[nx, ny] = solver(true_α, T0, dx, dy, dt, nt)

At each time step, we first update the interior of the domain using an
explicit Euler step with the spatial derivatives from ``heat_equation``,
then reset the four edges of ``T`` to zero. This enforces the Dirichlet
boundary conditions (``T1`` – ``T4``) at every step, ensuring the temperature
stays fixed at the boundary of the plate throughout the simulation.


Loss function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We measure the mean squared error between predicted and true final
temperature profiles:

.. code-block:: text

    def calculate_loss(α: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(α, T0, dx, dy, dt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss


Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are using Adam optimizer here:


.. math::

    \begin{align*}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
    \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
    \alpha &= \alpha - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align*}


.. code-block:: text


    def adam(α: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        α_new: ℝ = α - lr * m_hat / (sqrt(v_hat) + eps)
        return [α_new, m_new, v_new, t + 1.0]


Training
^^^^^^^^^^
We start with an initial guess of :math:`\alpha = 4.0` and train for 200 epochs
Once the training is finished, the :math:`\alpha` should be close to 2.0.


.. code-block:: text

    α: ℝ = 4.0

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.01

    epochs: ℕ = 200

    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, α)
        result = adam(α, g, m_adam, v_adam, t_adam, lr)
        α = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(α)


Visualize results
^^^^^^^^^^^^^^^^^^^^


.. code-block:: text

    pred_solution: ℝ[nx, ny] = solver(alpha, T0, dx, dy, dt, nt)
    visualize_trajectory_heat(true_solution, pred_solution, x, y)

.. note::
    Add ``visualize_trajectory_heat`` function in ``physika/runtime.py`` file:

    .. code-block:: python

        def visualize_trajectory_heat(true_solution, pred_solution, x, y):
            import matplotlib.pyplot as plt
            import numpy as np
            
            T_true_np = true_solution.detach().cpu().numpy()
            T_pred_np = pred_solution.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            X, Y = np.meshgrid(x_np, y_np, indexing='ij')

            # shared color scale so both plots are visually comparable
            vmin = min(T_true_np.min(), T_pred_np.min())
            vmax = max(T_true_np.max(), T_pred_np.max())

            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_surface(X, Y, T_true_np, cmap='gist_rainbow_r',
                            edgecolor='none', vmin=vmin, vmax=vmax)

            ax1.set_zlim(vmin, vmax)
            ax1.set_title('True Solution')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(X, Y, T_pred_np, cmap='gist_rainbow_r',
                            edgecolor='none', vmin=vmin, vmax=vmax)

            ax2.set_zlim(vmin, vmax)
            ax2.set_title('Predicted Solution')

            plt.tight_layout()
            plt.show()



.. figure:: /_static/tutorial_files/2d_pde/2d_heat.png
   :alt: Learned PDE trajectory vs ground truth
   :align: center
   :width: 700px

   Comparison between ground truth and learned trajectory after training.



Full code (2D heat equation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    # -------------------------------------
    # Helper functions
    # -------------------------------------

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results


    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results


    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x


    # -------------------------------------
    # Set Up the Domain
    # -------------------------------------

    true_α: ℝ = 2.0
    Lx: ℝ = 1.0
    Ly: ℝ = 1.0

    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 10

    dx: ℝ = Lx / (nx - 1)
    dy: ℝ = Ly / (ny - 1)


    # -------------------------------------
    # Time stepping
    # -------------------------------------

    fourier: ℝ = 0.49
    dt: ℝ = fourier / (1/dx**2 + 1/dy**2) / 10.0
    nt: ℝ = 100


    # -------------------------------------
    # Grid and initial condition
    # -------------------------------------

    # Boundary conditions
    T1: ℝ = 0
    T2: ℝ = 0
    T3: ℝ = 0
    T4: ℝ = 0

    T: ℝ[nx, ny] = zero_2d_array(nx, ny)

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)


    T0: ℝ[nx, ny] = zero_2d_array(nx, ny)
    for i:ℕ(0, nx):
        for j:ℕ(0, ny):
            T0[i, j] = exp(-20 * (((x[i]-0.5)**2) + ((y[j]-0.5)**2)))


    # -------------------------------------
    # Discretize the Heat equation
    # -------------------------------------


    def heat_equation(T: ℝ[m, n], dx: ℝ, dy: ℝ, α: ℝ): ℝ[m, n]:
        f: ℝ[m, n] = zero_2d_array(nx, ny)
        f[1:nx-1, 1:ny-1] = α * (
            ((T[0:nx-2, 1:ny-1] - 2 * T[1:nx-1, 1:ny-1] + T[2:nx, 1:ny-1]) / (dx**2)) +
            ((T[1:nx-1, 0:ny-2] - 2 * T[1:nx-1, 1:ny-1] + T[1:nx-1, 2:ny]) / (dy**2))
        )
        return f


    # -------------------------------------
    # Build the solver
    # -------------------------------------

    def solver(α: ℝ, T0: ℝ[m, n], dx: ℝ, dy: ℝ, dt: ℝ, nt: ℝ): ℝ[m, n]:
        T: ℝ[m, n] = T0
        for step:ℕ(0, nt):
            T = T + dt * heat_equation(T, dx, dy, α)
            T[:, 0] = 0
            T[:, ny-1] = 0
            T[0, :] = 0
            T[nx-1, :] = 0
        return T

    true_solution: ℝ[nx, ny] = solver(true_α, T0, dx, dy, dt, nt)

    # -------------------------------------
    # Loss function
    # -------------------------------------


    def calculate_loss(α: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(α, T0, dx, dy, dt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss

    # -------------------------------------
    # Optimizer
    # -------------------------------------

    def adam(α: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        α_new: ℝ = α - lr * m_hat / (sqrt(v_hat) + eps)
        return [α_new, m_new, v_new, t + 1.0]


    # -------------------------------------
    # Training
    # -------------------------------------


    α: ℝ = 4.0

    guess_solution: ℝ[nx, ny] = solver(α, T0, dx, dy, dt, nt)
    #visualize_trajectory_heat(true_solution, guess_solution, x, y)

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.01

    epochs: ℕ = 200

    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, α)
        result = adam(α, g, m_adam, v_adam, t_adam, lr)
        α = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(α)

    # α should be closer to 2.0
    α

    pred_solution: ℝ[nx, ny] = solver(α, T0, dx, dy, dt, nt)
    visualize_trajectory_heat(true_solution, pred_solution, x, y)





2D Wave equation
----------------

.. math::

   \frac{\partial^2 u}{\partial t^2}
   =
   c^2\left(
   \frac{\partial^2 u}{\partial x^2}
   +
   \frac{\partial^2 u}{\partial y^2}
   \right)

where :math:`u(x, y, t)` is the displacement field, :math:`x` and :math:`y`
are the spatial coordinates, :math:`t` is time, and :math:`c` is the wave
speed, the parameter we want to learn.


Helper functions
^^^^^^^^^^^^^^^^


.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols:ℝ ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x


Set Up the Domain
^^^^^^^^^^^^^^^^^

.. code-block:: text

    Lx: ℝ = 1.0
    Ly: ℝ = 1.0
    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 2.0

    dx: ℝ = Lx / (nx - 1)
    dy: ℝ = Ly / (ny - 1)

    true_c: ℝ = 1.0

``Lx`` and ``Ly`` define the size of the rectangular domain, while ``nx`` and
``ny`` specify the number of grid points along the :math:`x` and
:math:`y` directions. The corresponding spatial grid spacings are
``dx`` and ``dy``.


Time stepping
^^^^^^^^^^^^^

.. code-block:: text

    cfl: ℝ = 0.4
    dt: ℝ = cfl / (5.0 * sqrt(1/dx**2 + 1/dy**2))
    nt: ℝ = 50

The time step ``dt`` is chosen using the CFL number to satisfy the stability
condition for the explicit finite-difference wave solver. ``nt`` specifies the
total number of time steps used in the simulation.


Grid and initial condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)

    pi: ℝ = 3.14
    u0: ℝ[nx,ny] = zero_2d_array(nx, ny)

    for i:ℕ(0, nx):
        for j:ℕ(0, ny):
            u0[i,j] = sin(2 * pi * x[i]) * sin(pi * y[j])

    v0: ℝ[nx,ny] = zero_2d_array(nx, ny)

The spatial grid is constructed using ``x`` and ``y``. The initial
displacement ``u0`` is initialized as a smooth standing-wave profile using a
product of sine functions, while the initial velocity ``v0`` is set to zero.
These define the initial conditions from which the wave evolves over time.


Discretize the Wave equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using second-order central differences for the spatial derivatives, the 2D
wave equation becomes:

.. math::

    \begin{align*}
    f_{i,j}
    =
    c^2\left(
    \frac{u_{i-1,j}-2u_{i,j}+u_{i+1,j}}{\Delta x^2}
    +
    \frac{u_{i,j-1}-2u_{i,j}+u_{i,j+1}}{\Delta y^2}
    \right)
    \end{align*}

.. code-block:: text

    def wave_equation(u: ℝ[m, n], dx: ℝ, dy: ℝ, c: ℝ): ℝ[m, n]:
        lap: ℝ[m, n] = zero_2d_array(nx, ny)
        lap[1:nx-1, 1:ny-1] = c**2 * (
            ((u[0:nx-2, 1:ny-1] - 2 * u[1:nx-1, 1:ny-1] + u[2:nx, 1:ny-1]) / (dx**2)) +
            ((u[1:nx-1, 0:ny-2] - 2 * u[1:nx-1, 1:ny-1] + u[1:nx-1, 2:ny]) / (dy**2))
        )
        return lap

The ``wave_equation`` function computes the spatial Laplacian of the current
displacement field using second-order central differences. Multiplying by
:math:`c^2` gives the acceleration term used by the wave equation.


Build the solver
^^^^^^^^^^^^^^^^

.. code-block:: text

    def solver(c: ℝ, u0: ℝ[m, n], v0: ℝ[m, n], dx: ℝ, dy: ℝ, dt: ℝ, nt: ℝ): ℝ[m, n]:
        u_prev: ℝ[m, n] = u0
        u_curr: ℝ[m, n] = u0
        for step:ℕ(0, nt):
            accel = wave_equation(u_curr, dx, dy, c)
            u_next = 2 * u_curr - u_prev + dt**2 * accel
            u_next[:, 0] = 0
            u_next[:, ny-1] = 0
            u_next[0, :] = 0
            u_next[nx-1, :] = 0
            u_prev = u_curr
            u_curr = u_next
        return u_curr

    true_solution: ℝ[nx, ny] = solver(true_c, u0, v0, dx, dy, dt, nt)

The solver advances the solution using a second-order finite-difference
time-stepping scheme. At each step, the acceleration is computed from the
current displacement, the next displacement is updated using the current and
previous states, and the boundary values are reset to zero. This enforces
homogeneous Dirichlet boundary conditions throughout the simulation.


Loss function
^^^^^^^^^^^^^

We measure the mean squared error between the predicted and true final
displacement fields:

.. code-block:: text

    def calculate_loss(c: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(c, u0, v0, dx, dy, dt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss


Optimizer
^^^^^^^^^

We are using Adam optimizer here:

.. math::

    \begin{align*}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
    \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
    c &= c - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align*}

.. code-block:: text

    def adam(c: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        c_new: ℝ = c - lr * m_hat / (sqrt(v_hat) + eps)
        return [c_new, m_new, v_new, t + 1.0]


Training
^^^^^^^^

We start with an initial guess of :math:`c = 2.0` and train for 10 epochs:

.. code-block:: text

    c: R = 2.0

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.1

    epochs: ℕ = 10

    for i:N(epochs):
        physika_print(i)
        g = grad(calculate_loss, c)
        result = adam(c, g, m_adam, v_adam, t_adam, lr)
        c = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(c)


Visualize results
^^^^^^^^^^^^^^^^^

.. code-block:: text

    pred_solution = solver(c, u0, v0, dx, dy, dt, nt)
    visualize_trajectory(true_solution, pred_solution, x, y)

.. figure:: /_static/tutorial_files/2d_pde/2d_wave.png
   :alt: Learned PDE trajectory vs ground truth
   :align: center
   :width: 700px

   Comparison between the ground truth and learned wave field after training.



Full code (2D wave equation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    # -------------------------------------
    # Helper functions
    # -------------------------------------


    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols:ℝ ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x

    # -------------------------------------
    # Set up the domain
    # -------------------------------------

    Lx: ℝ = 1.0
    Ly: ℝ = 1.0
    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 2.0

    dx: ℝ = Lx / (nx - 1)
    dy: ℝ = Ly / (ny - 1)

    true_c: ℝ = 1.0

    # -------------------------------------
    # Time stepping
    # -------------------------------------

    cfl: ℝ = 0.4
    dt: ℝ = cfl / (5.0 * sqrt(1/dx**2 + 1/dy**2))
    nt: ℝ = 50

    # -------------------------------------
    # Grid and Initial condition
    # -------------------------------------

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)

    pi: ℝ = 3.14
    u0: ℝ[nx,ny] = zero_2d_array(nx, ny)

    for i:ℕ(0, nx):
        for j:ℕ(0, ny):
            u0[i,j] = sin(2 * pi * x[i]) * sin(pi * y[j])

    v0: ℝ[nx,ny] = zero_2d_array(nx, ny)

    # -------------------------------------
    # Discretize the wave equation
    # -------------------------------------


    def wave_equation(u: ℝ[m, n], dx: ℝ, dy: ℝ, c: ℝ): ℝ[m, n]:
        lap: ℝ[m, n] = zero_2d_array(nx, ny)
        lap[1:nx-1, 1:ny-1] = c**2 * (
            ((u[0:nx-2, 1:ny-1] - 2 * u[1:nx-1, 1:ny-1] + u[2:nx, 1:ny-1]) / (dx**2)) +
            ((u[1:nx-1, 0:ny-2] - 2 * u[1:nx-1, 1:ny-1] + u[1:nx-1, 2:ny]) / (dy**2))
        )
        return lap

    # -------------------------------------
    # Build the solver
    # -------------------------------------

    def solver(c: ℝ, u0: ℝ[m, n], v0: ℝ[m, n], dx: ℝ, dy: ℝ, dt: ℝ, nt: ℝ): ℝ[m, n]:
        u_prev: ℝ[m, n] = u0
        u_curr: ℝ[m, n] = u0
        for step:ℕ(0, nt):
            accel = wave_equation(u_curr, dx, dy, c)
            u_next = 2 * u_curr - u_prev + dt**2 * accel
            u_next[:, 0] = 0
            u_next[:, ny-1] = 0
            u_next[0, :] = 0
            u_next[nx-1, :] = 0
            u_prev = u_curr
            u_curr = u_next
        return u_curr

    true_solution: ℝ[nx, ny] = solver(true_c, u0, v0, dx, dy, dt, nt)


    # -------------------------------------
    # Loss function
    # -------------------------------------

    def calculate_loss(c: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(c, u0, v0, dx, dy, dt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss

    # -------------------------------------
    # Optimizer
    # -------------------------------------

    def adam(c: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        c_new: ℝ = c - lr * m_hat / (sqrt(v_hat) + eps)
        return [c_new, m_new, v_new, t + 1.0]

    # -------------------------------------
    # Training
    # -------------------------------------

    c: ℝ = 3.0


    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.01

    epochs: ℕ = 400


    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, c)
        result = adam(c, g, m_adam, v_adam, t_adam, lr)
        c = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(c)

    pred_solution: ℝ[nx, ny] = solver(c, u0, v0, dx, dy, dt, nt)
    visualize_trajectory(true_solution, pred_solution, x, y)
