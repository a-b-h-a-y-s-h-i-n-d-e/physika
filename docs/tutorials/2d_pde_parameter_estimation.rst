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
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
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

    Δx: ℝ = Lx / (nx - 1)
    Δy: ℝ = Ly / (ny - 1)

``Lx`` and ``Ly`` define the length on both the axis, ``nx`` and ``ny`` set the
number of grid points along each axis, and ``Δx`` and ``Δy`` are the
resulting grid spacings.


Time stepping
^^^^^^^^^^^^^

.. code-block::

    fourier: ℝ = 0.49
    Δt: ℝ = fourier / (1/Δx**2 + 1/Δy**2) / 10.0
    nt: ℝ = 100

.. note::
   For the 2D heat equation, the ``CFL stability condition`` for explicit time-stepping is defined as:

   .. math::

      \alpha \, \Delta t \left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right) \leq 0.5

   Here we are using a Fourier number of ``0.49``, with an
   additional safety factor of ``10`` for extra margin.

The time step ``Δt`` is computed from the CFL number [SimScaleCFL]_ to satisfy the
stability condition for explicit time-stepping, and ``nt`` is the total
number of time steps to simulate.


Grid and initial condition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    # Boundary conditions
    T1: ℝ, T2: ℝ, T3: ℝ, T4: ℝ = 0, 0, 0, 0

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)

    T0: R[nx, ny] = for i:N(nx) → for j:N(ny) → exp(-20 * (((x[i]-0.5)**2) + ((y[j]-0.5)**2)))

``T1``–``T4`` set the (zero) Dirichlet boundary conditions on the four edges
of the plate. We then build the spatial grid using ``x`` and ``y``, and
initialize the temperature field ``T0`` as a Gaussian pulse centered at the
middle of the domain, this is the initial condition the PDE solver will evolve
forward in time.



Discretize the Heat equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using second-order central differences for the spatial derivatives, the 2D
heat equation becomes [CadenceHeat2D]_ :

.. math::

    \begin{align*}
    f_{i,j} = \alpha \left(
    \frac{T_{i-1,j} - 2T_{i,j} + T_{i+1,j}}{\Delta x^2} +
    \frac{T_{i,j-1} - 2T_{i,j} + T_{i,j+1}}{\Delta y^2}
    \right)
    \end{align*}

.. code-block:: text

    def heat_equation(T: ℝ[m, n], Δx: ℝ, Δy: ℝ, α: ℝ): ℝ[m, n]:
        f: ℝ[m, n] = zero_2d_array(nx, ny)
        f[1:nx-1, 1:ny-1] = α * (
            ((T[0:nx-2, 1:ny-1] - 2 * T[1:nx-1, 1:ny-1] + T[2:nx, 1:ny-1]) / (Δx**2)) +
            ((T[1:nx-1, 0:ny-2] - 2 * T[1:nx-1, 1:ny-1] + T[1:nx-1, 2:ny]) / (Δy**2))
        )
        return f

.. note::

   The code above uses a vectorized (slice-based) implementation for
   efficiency. If you find explicit loops easier to follow, here's an
   equivalent, more readable version using nested ``for`` loops instead:

   .. code-block:: text

       def heat_equation(T: ℝ[m, n], Δx: ℝ, Δy: ℝ, α: ℝ): ℝ[m, n]:
           f: ℝ[m, n] = zero_2d_array(nx, ny)
           for i:ℕ(1, nx-1):
               for j:ℕ(1, ny-1):
                   f[i, j] = α * (
                       ((T[i-1, j] - 2 * T[i, j] + T[i+1, j]) / (Δx**2)) +
                       ((T[i, j-1] - 2 * T[i, j] + T[i, j+1]) / (Δy**2))
                   )
           return f

   Both versions compute the same result — the vectorized form is faster
   since it avoids explicit Python-level looping over every grid point,
   while the loop-based form more directly mirrors the mathematical
   stencil shown above.


Build the solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: text

    def solver(α: ℝ, T0: ℝ[m, n], Δx: ℝ, Δy: ℝ, Δt: ℝ, nt: ℝ): ℝ[m, n]:
        T: ℝ[m, n] = T0
        for step:ℕ(0, nt):
            T = T + Δt * heat_equation(T, Δx, Δy, α)
            T[:, 0] = 0
            T[:, ny-1] = 0
            T[0, :] = 0
            T[nx-1, :] = 0
        return T

    true_solution: ℝ[nx, ny] = solver(true_α, T0, Δx, Δy, Δt, nt)

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
        predictions: ℝ[nx, ny] = solver(α, T0, Δx, Δy, Δt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss


Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are using Adam optimizer here [KingmaBa2014]_ :


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

    pred_solution: ℝ[nx, ny] = solver(alpha, T0, Δx, Δy, Δt, nt)
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



.. figure:: /_static/tutorial_files/2d_pde/2d_heat_results.png
   :alt: Learned PDE trajectory vs ground truth
   :align: center
   :width: 700px
   :name: fig-2d-heat-results

   Figure 1: Comparison between ground truth and learned trajectory after training.



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
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
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

    Δx: ℝ = Lx / (nx - 1)
    Δy: ℝ = Ly / (ny - 1)


    # -------------------------------------
    # Time stepping
    # -------------------------------------

    fourier: ℝ = 0.49
    Δt: ℝ = fourier / (1/Δx**2 + 1/Δy**2) / 10.0
    nt: ℝ = 100


    # -------------------------------------
    # Grid and initial condition
    # -------------------------------------

    # Boundary conditions
    T1: ℝ, T2: ℝ, T3: ℝ, T4: ℝ = 0, 0, 0, 0

    x: ℝ[nx] = linspace(0, Lx, nx)
    y: ℝ[ny] = linspace(0, Ly, ny)

    T0: R[nx, ny] = for i:N(nx) → for j:N(ny) → exp(-20 * (((x[i]-0.5)**2) + ((y[j]-0.5)**2)))


    # -------------------------------------
    # Discretize the Heat equation
    # -------------------------------------


    def heat_equation(T: ℝ[m, n], Δx: ℝ, Δy: ℝ, α: ℝ): ℝ[m, n]:
        f: ℝ[m, n] = zero_2d_array(nx, ny)
        f[1:nx-1, 1:ny-1] = α * (
            ((T[0:nx-2, 1:ny-1] - 2 * T[1:nx-1, 1:ny-1] + T[2:nx, 1:ny-1]) / (Δx**2)) +
            ((T[1:nx-1, 0:ny-2] - 2 * T[1:nx-1, 1:ny-1] + T[1:nx-1, 2:ny]) / (Δy**2))
        )
        return f


    # -------------------------------------
    # Build the solver
    # -------------------------------------

    def solver(α: ℝ, T0: ℝ[m, n], Δx: ℝ, Δy: ℝ, Δt: ℝ, nt: ℝ): ℝ[m, n]:
        T: ℝ[m, n] = T0
        for step:ℕ(0, nt):
            T = T + Δt * heat_equation(T, Δx, Δy, α)
            T[:, 0] = 0
            T[:, ny-1] = 0
            T[0, :] = 0
            T[nx-1, :] = 0
        return T

    true_solution: ℝ[nx, ny] = solver(true_α, T0, Δx, Δy, Δt, nt)

    # -------------------------------------
    # Loss function
    # -------------------------------------


    def calculate_loss(α: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(α, T0, Δx, Δy, Δt, nt)
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

    guess_solution: ℝ[nx, ny] = solver(α, T0, Δx, Δy, Δt, nt)
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

    pred_solution: ℝ[nx, ny] = solver(α, T0, Δx, Δy, Δt, nt)
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



Set Up the Domain
^^^^^^^^^^^^^^^^^

.. code-block:: text

    Lx: ℝ = 1.0
    Ly: ℝ = 1.0
    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 2.0

    Δx: ℝ = Lx / (nx - 1)
    Δy: ℝ = Ly / (ny - 1)

    true_c: ℝ = 1.0

``Lx`` and ``Ly`` define the size of the rectangular domain, while ``nx`` and
``ny`` specify the number of grid points along the :math:`x` and
:math:`y` directions. The corresponding spatial grid spacings are
``Δx`` and ``Δy``.


Time stepping
^^^^^^^^^^^^^

.. code-block:: text

    cfl: ℝ = 0.4
    Δt: ℝ = cfl / (5.0 * sqrt(1/Δx**2 + 1/Δy**2))
    nt: ℝ = 50

.. note::
   For the 2D wave equation, the ``CFL stability condition`` for explicit time-stepping is defined as:

   .. math::

      c \, \Delta t \, \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}} \leq 1

   where ``c`` is the wave speed. Here we are using  Fourier number of ``0.4``, with an additional safety factor of ``5``
   for extra margin.

The time step ``Δt`` is chosen using the CFL number [SimScaleCFL]_ to satisfy the stability
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
wave equation becomes [AdamsWave2D]_ :

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

    def wave_equation(u: ℝ[m, n], Δx: ℝ, Δy: ℝ, c: ℝ): ℝ[m, n]:
        lap: ℝ[m, n] = zero_2d_array(nx, ny)
        lap[1:nx-1, 1:ny-1] = c**2 * (
            ((u[0:nx-2, 1:ny-1] - 2 * u[1:nx-1, 1:ny-1] + u[2:nx, 1:ny-1]) / (Δx**2)) +
            ((u[1:nx-1, 0:ny-2] - 2 * u[1:nx-1, 1:ny-1] + u[1:nx-1, 2:ny]) / (Δy**2))
        )
        return lap

The ``wave_equation`` function computes the spatial Laplacian of the current
displacement field using second-order central differences. Multiplying by
:math:`c^2` gives the acceleration term used by the wave equation.

.. note::

   The code above uses a vectorized (slice-based) implementation for
   efficiency. If you find explicit loops easier to follow, here's an
   equivalent, more readable version using nested ``for`` loops instead:

   .. code-block:: text

        def wave_equation(u: ℝ[m, n], Δx: ℝ, Δy: ℝ, c: ℝ): ℝ[m, n]:
            lap: ℝ[m, n] = zero_2d_array(nx, ny)
            for i:ℕ(1, nx-1):
                for j:ℕ(1, ny-1):
                    lap[i, j] = c**2 * (
                        ((u[i-1,j] - 2*u[i,j] + u[i+1,j]) / (Δx**2)) +
                        ((u[i,j-1] - 2*u[i,j] + u[i,j+1]) / (Δy**2))
                    )
            return lap

   Both versions compute the same result — the vectorized form is faster
   since it avoids explicit Python-level looping over every grid point,
   while the loop-based form more directly mirrors the mathematical
   stencil shown above.


Build the solver
^^^^^^^^^^^^^^^^

.. code-block:: text

    def solver(c: ℝ, u0: ℝ[m, n], v0: ℝ[m, n], Δx: ℝ, Δy: ℝ, Δt: ℝ, nt: ℝ): ℝ[m, n]:
        u_prev: ℝ[m, n] = u0
        u_curr: ℝ[m, n] = u0
        for step:ℕ(0, nt):
            accel = wave_equation(u_curr, Δx, Δy, c)
            u_next = 2 * u_curr - u_prev + Δt**2 * accel
            u_next[:, 0] = 0
            u_next[:, ny-1] = 0
            u_next[0, :] = 0
            u_next[nx-1, :] = 0
            u_prev = u_curr
            u_curr = u_next
        return u_curr

    true_solution: ℝ[nx, ny] = solver(true_c, u0, v0, Δx, Δy, Δt, nt)

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
        predictions: ℝ[nx, ny] = solver(c, u0, v0, Δx, Δy, Δt, nt)
        diff: ℝ[nx, ny] = predictions - true_solution
        loss: ℝ = mean(diff**2)
        return loss


Optimizer
^^^^^^^^^

We are using Adam optimizer here [KingmaBa2014]_ :

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

    pred_solution: ℝ[nx, ny] = solver(c, u0, v0, Δx, Δy, Δt, nt)
    visualize_trajectory_wave(true_solution, pred_solution, x, y)

.. note::
    Add ``visualize_trajectory_wave`` function in ``physika/runtime.py`` file:

    .. code-block:: python

        def visualize_trajectory_wave(true_solution, pred_solution, x, y):
            u_true = true_solution.detach().cpu().numpy()
            u_pred = pred_solution.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            # shared z-limits so both plots are visually comparable
            vmax = max(np.max(np.abs(u_true)), np.max(np.abs(u_pred)))
            vmin = -vmax

            X, Y = np.meshgrid(x_np, y_np, indexing='ij')

            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_wireframe(X, Y, u_true, color='steelblue', linewidth=0.7, rstride=1, cstride=1)
            ax1.set_zlim(vmin, vmax)
            ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('u')
            ax1.set_title('True Solution')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_wireframe(X, Y, u_pred, color='indianred', linewidth=0.7, rstride=1, cstride=1)
            ax2.set_zlim(vmin, vmax)
            ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]'); ax2.set_zlabel('u')
            ax2.set_title('Predicted Solution')

            plt.tight_layout()
            plt.show()

.. figure:: /_static/tutorial_files/2d_pde/2d_wave_results.png
   :alt: Learned PDE trajectory vs ground truth
   :align: center
   :width: 700px
   :name: fig-2d-wave-results

   Figure 2: Comparison between the ground truth and learned wave field after training.



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
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
        return x

    # -------------------------------------
    # Set up the domain
    # -------------------------------------

    Lx: ℝ = 1.0
    Ly: ℝ = 1.0
    nx: ℝ = 40
    ny: ℝ = 40
    tf: ℝ = 2.0

    Δx: ℝ = Lx / (nx - 1)
    Δy: ℝ = Ly / (ny - 1)

    true_c: ℝ = 1.0

    # -------------------------------------
    # Time stepping
    # -------------------------------------

    cfl: ℝ = 0.4
    Δt: ℝ = cfl / (5.0 * sqrt(1/Δx**2 + 1/Δy**2))
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


    def wave_equation(u: ℝ[m, n], Δx: ℝ, Δy: ℝ, c: ℝ): ℝ[m, n]:
        lap: ℝ[m, n] = zero_2d_array(nx, ny)
        lap[1:nx-1, 1:ny-1] = c**2 * (
            ((u[0:nx-2, 1:ny-1] - 2 * u[1:nx-1, 1:ny-1] + u[2:nx, 1:ny-1]) / (Δx**2)) +
            ((u[1:nx-1, 0:ny-2] - 2 * u[1:nx-1, 1:ny-1] + u[1:nx-1, 2:ny]) / (Δy**2))
        )
        return lap

    # -------------------------------------
    # Build the solver
    # -------------------------------------

    def solver(c: ℝ, u0: ℝ[m, n], v0: ℝ[m, n], Δx: ℝ, Δy: ℝ, Δt: ℝ, nt: ℝ): ℝ[m, n]:
        u_prev: ℝ[m, n] = u0
        u_curr: ℝ[m, n] = u0
        for step:ℕ(0, nt):
            accel = wave_equation(u_curr, Δx, Δy, c)
            u_next = 2 * u_curr - u_prev + Δt**2 * accel
            u_next[:, 0] = 0
            u_next[:, ny-1] = 0
            u_next[0, :] = 0
            u_next[nx-1, :] = 0
            u_prev = u_curr
            u_curr = u_next
        return u_curr

    true_solution: ℝ[nx, ny] = solver(true_c, u0, v0, Δx, Δy, Δt, nt)


    # -------------------------------------
    # Loss function
    # -------------------------------------

    def calculate_loss(c: ℝ): ℝ:
        predictions: ℝ[nx, ny] = solver(c, u0, v0, Δx, Δy, Δt, nt)
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

    pred_solution: ℝ[nx, ny] = solver(c, u0, v0, Δx, Δy, Δt, nt)
    visualize_trajectory_wave(true_solution, pred_solution, x, y)





2D Navier-Stokes equation (Lid-driven cavity)
---------------------------------------------

The 2D incompressible Navier-Stokes equations are [NavierStokesWiki]_ :

.. math::

    \begin{align*}
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}
    &= -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
    \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}
    &= -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) \\
    \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} &= 0
    \end{align*}

where,

- :math:`u(x, y, t)` and :math:`v(x, y, t)` are the horizontal and vertical
  velocity components
- :math:`p(x, y, t)` is the pressure
- :math:`\nu` is the kinematic viscosity
- :math:`\rho` is the fluid density, the parameter we want to learn

The third equation (continuity equation) of Navier-stokes represents incompressibility property of the fluid. As the fluid flow is in
a steady-state, the field properties are not functions of time and the equation reduces to one comprising of velocity vector.


Problem setup
^^^^^^^^^^^^^^^

.. code-block:: text

    n_points: ℝ = 21
    domain_size: ℝ = 1.0
    n_iterations: ℝ = 500

    time_step_length: ℝ = 0.001
    ν: ℝ = 0.1
    true_ρ: ℝ = 1.0
    horizontal_velocity_top: ℝ = 1.0

    n_pressure_poisson_iterations: ℝ = 10
    stability_safety_factor: ℝ = 0.5

    element_length: ℝ = domain_size / (n_points - 1)

    x: ℝ[n_points] = linspace(0.0, domain_size, n_points)
    y: ℝ[n_points] = linspace(0.0, domain_size, n_points)

- ``domain_size`` is the length of the (square) cavity
- ``n_points`` is the number of grid points along each axis (for both x and y axis)
- ``element_length`` is the grid spacing (used as both :math:`\Delta x` and
  :math:`\Delta y` since the grid is uniform)
- ``time_step_length`` is :math:`\Delta t`, which is size of each step
- ``ν`` is :math:`\nu` also known as kinematic viscosity
- ``density`` is :math:`\rho` also known as density (the parameter we want to learn)
- ``horizontal_velocity_top`` is the lid velocity that drives the flow
- ``n_pressure_poisson_iterations`` is the number of passes we run at each time step to solve pressure field.


Discretize the spatial derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before assembling the equations, we need discrete approximations of the
first derivatives (central difference) and the Laplacian, which are reused
throughout the solver.

.. math::

    \begin{align*}
    \frac{\partial f}{\partial x}\bigg|_{i,j} &\approx \frac{f_{i,j+1} - f_{i,j-1}}{2\Delta x} \\
    \frac{\partial f}{\partial y}\bigg|_{i,j} &\approx \frac{f_{i+1,j} - f_{i-1,j}}{2\Delta y} \\
    \nabla^2 f \big|_{i,j} &\approx \frac{f_{i,j-1} + f_{i-1,j} + f_{i,j+1} + f_{i+1,j} - 4f_{i,j}}{\Delta x^2}
    \end{align*}

.. code-block:: text

    def central_difference_x(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[1:n_points-1, 2:n_points] -
            f[1:n_points-1, 0:n_points-2]
        ) / (2 * element_length)
        return diff

    def central_difference_y(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[2:n_points, 1:n_points-1] -
            f[0:n_points-2, 1:n_points-1]
        ) / (2 * element_length)
        return diff

    def laplace(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[1:n_points-1, 0:n_points-2] +   # left
            f[0:n_points-2, 1:n_points-1] +   # up
            f[1:n_points-1, 2:n_points] +     # right
            f[2:n_points, 1:n_points-1] -     # down
            4 * f[1:n_points-1, 1:n_points-1]
        ) / (element_length ** 2)
        return diff

Build the solver
^^^^^^^^^^^^^^^^^^^^^^^^

Unlike the heat and wave equations, where the solver is a single
straightforward update step, the Navier-Stokes solver involves several
coupled steps. We'll walk through each one in detail below before
assembling them into the full solver. [ANLINSChorin]_ [CadenceNavierStokes]_  



Prediction step for the velocity field 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The projection method first computes a *tentative* velocity field by
advancing the momentum equations while ignoring the pressure gradient term:

.. math::

    \begin{align*}
    u^{*} &= u^{n} + \Delta t \left(-\left(u^{n}\frac{\partial u^{n}}{\partial x} + v^{n}\frac{\partial u^{n}}{\partial y}\right) + \nu \nabla^2 u^{n}\right) \\
    v^{*} &= v^{n} + \Delta t \left(-\left(u^{n}\frac{\partial v^{n}}{\partial x} + v^{n}\frac{\partial v^{n}}{\partial y}\right) + \nu \nabla^2 v^{n}\right)
    \end{align*}

Here :math:`u^{*}, v^{*}` denote the tentative velocities, which don't yet
satisfy the incompressibility constraint which gets corrected in the pressure
projection step below.

.. code-block:: text

    d_u_prev__d_x = central_difference_x(u_prev)
    d_u_prev__d_y = central_difference_y(u_prev)
    d_v_prev__d_x = central_difference_x(v_prev)
    d_v_prev__d_y = central_difference_y(v_prev)
    laplace__u_prev = laplace(u_prev)
    laplace__v_prev = laplace(v_prev)
    u_tent = u_prev + time_step_length * (
        - (
            u_prev * d_u_prev__d_x + v_prev * d_u_prev__d_y
        ) + ν * laplace__u_prev
    )
    v_tent = v_prev + time_step_length * (
        - (
            u_prev * d_v_prev__d_x + v_prev * d_v_prev__d_y
        ) + ν * laplace__v_prev
    )


After this we update the velocity boundary values.
(Homogeneous Dirichlet BC everywhere except for the horizontal velocity at the top)


.. math::

    \begin{align*}
    u = 0, \; v = 0 &\quad \text{on left, right, bottom walls} \\
    u = u_{\text{lid}}, \; v = 0 &\quad \text{on top wall}
    \end{align*}

.. code-block:: text

    u_tent[0, :] = 0.0
    u_tent[-1, :] = horizontal_velocity_top
    u_tent[:, 0] = 0.0
    u_tent[:, -1] = 0.0
    v_tent[0, :] = 0.0
    v_tent[-1, :] = 0.0
    v_tent[:, 0] = 0.0
    v_tent[:, -1] = 0.0


Correction step for the pressure field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The predicted velocity field is then used to compute a provisional pressure field using a Poisson equation derived from the incompressibility constraint.
The pressure correction is applied to remove any divergence from the predicted velocity field.

.. math::

    \nabla^2 p = \frac{\rho}{\Delta t}\left(\frac{\partial u^{*}}{\partial x} + \frac{\partial v^{*}}{\partial y}\right)


.. code-block:: text

    d_u_tent__d_x = central_difference_x(u_tent)
    d_v_tent__d_y = central_difference_y(v_tent)
    rhs = (ρ / time_step_length * (d_u_tent__d_x + d_v_tent__d_y))
    for k:N(n_pressure_poisson_iterations):
        p_next = zero_2d_array(n_points, n_points)
        p_next[1:-1, 1:-1] = 0.25 * (
            p_prev[1:-1, :-2] +
            p_prev[:-2, 1:-1] +
            p_prev[1:-1, 2:] +
            p_prev[2:, 1:-1] -
            element_length**2 * rhs[1:-1, 1:-1]
        )

After this again we will update boundary values, where 
we use homogeneous Neumann conditions on the left, right, and bottom walls, and fix :math:`p = 0` on the top wall:


.. code-block:: text

    p_next[:, -1] = p_next[:, -2]
    p_next[0, :] = p_next[1, :]
    p_next[:, 0] = p_next[:, 1]
    p_next[-1, :] = 0.0
    p_prev = p_next


Velocity correction (projection step)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the corrected pressure field in hand, we project the tentative
velocity onto its divergence-free component:

.. math::

    \begin{align*}
    u^{n+1} &= u^{*} - \frac{\Delta t}{\rho}\frac{\partial p}{\partial x} \\
    v^{n+1} &= v^{*} - \frac{\Delta t}{\rho}\frac{\partial p}{\partial y}
    \end{align*}

.. code-block:: text

    d_p_next__d_x = central_difference_x(p_next)
    d_p_next__d_y = central_difference_y(p_next)
    u_next = (
        u_tent -
        time_step_length / ρ *
        d_p_next__d_x
    )
    v_next = (
        v_tent -
        time_step_length / ρ *
        d_p_next__d_y
    )
    u_next[0, :] = 0.0
    u_next[:, 0] = 0.0
    u_next[:, -1] = 0.0
    u_next[-1, :] = horizontal_velocity_top
    v_next[0, :] = 0.0
    v_next[:, 0] = 0.0
    v_next[:, -1] = 0.0
    v_next[-1, :] = 0.0


Wrapping everything in solver function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Putting it all together, the solver repeats the tentative velocity step,
pressure Poisson solve, and velocity correction for ``n_iterations`` time
steps:

.. code-block:: text

    def solver(ρ: R): R[3, m, n]:
        u_prev = zero_2d_array(n_points, n_points)
        v_prev = zero_2d_array(n_points, n_points)
        p_prev = zero_2d_array(n_points, n_points)
        for i:N(n_iterations):
            d_u_prev__d_x = central_difference_x(u_prev)
            d_u_prev__d_y = central_difference_y(u_prev)
            d_v_prev__d_x = central_difference_x(v_prev)
            d_v_prev__d_y = central_difference_y(v_prev)
            laplace__u_prev = laplace(u_prev)
            laplace__v_prev = laplace(v_prev)
            u_tent = u_prev + time_step_length * (
                - (
                    u_prev * d_u_prev__d_x + v_prev * d_u_prev__d_y
                ) + ν * laplace__u_prev
            )
            v_tent = v_prev + time_step_length * (
                - (
                    u_prev * d_v_prev__d_x + v_prev * d_v_prev__d_y
                ) + ν * laplace__v_prev
            )
            u_tent[0, :] = 0.0
            u_tent[-1, :] = horizontal_velocity_top
            u_tent[:, 0] = 0.0
            u_tent[:, -1] = 0.0
            v_tent[0, :] = 0.0
            v_tent[-1, :] = 0.0
            v_tent[:, 0] = 0.0
            v_tent[:, -1] = 0.0
            d_u_tent__d_x = central_difference_x(u_tent)
            d_v_tent__d_y = central_difference_y(v_tent)
            rhs = (ρ / time_step_length * (d_u_tent__d_x + d_v_tent__d_y))
            for k:N(n_pressure_poisson_iterations):
                p_next = zero_2d_array(n_points, n_points)
                p_next[1:-1, 1:-1] = 0.25 * (
                    p_prev[1:-1, :-2] +
                    p_prev[:-2, 1:-1] +
                    p_prev[1:-1, 2:] +
                    p_prev[2:, 1:-1] -
                    element_length**2 * rhs[1:-1, 1:-1]
                )
                p_next[:, -1] = p_next[:, -2]
                p_next[0, :] = p_next[1, :]
                p_next[:, 0] = p_next[:, 1]
                p_next[-1, :] = 0.0
                p_prev = p_next
            d_p_next__d_x = central_difference_x(p_next)
            d_p_next__d_y = central_difference_y(p_next)
            u_next = (
                u_tent -
                time_step_length / ρ *
                d_p_next__d_x
            )
            v_next = (
                v_tent -
                time_step_length / ρ *
                d_p_next__d_y
            )
            u_next[0, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
            u_next[-1, :] = horizontal_velocity_top
            v_next[0, :] = 0.0
            v_next[:, 0] = 0.0
            v_next[:, -1] = 0.0
            v_next[-1, :] = 0.0
            u_prev = u_next
            v_prev = v_next
            p_prev = p_next
        return [u_prev, v_prev, p_prev]



Learning the density
^^^^^^^^^^^^^^^^^^^^^^^^

We generate a reference (ground-truth) solution using the true density,
then define a loss function that measures how far a candidate density's
solution is from that reference:

.. math::

    \mathcal{L}(\rho) = \text{mean}\left((u_{\rho} - u_{\text{true}})^2\right) +
                        \text{mean}\left((v_{\rho} - v_{\text{true}})^2\right) +
                        \text{mean}\left((p_{\rho} - p_{\text{true}})^2\right)

.. code-block:: text

    true_solution: ℝ[3, n_points, n_points] = solver(true_ρ)
    true_u: ℝ[n_points, n_points] = true_solution[0]
    true_v: ℝ[n_points, n_points] = true_solution[1]
    true_p: ℝ[n_points, n_points] = true_solution[2]

    def calculate_loss(ρ: ℝ): ℝ:
        predictions: ℝ[3, n_points, n_points] = solver(ρ)
        pred_u: ℝ[n_points, n_points] = predictions[0]
        pred_v: ℝ[n_points, n_points] = predictions[1]
        pred_p: ℝ[n_points, n_points] = predictions[2]
        loss_u: ℝ = mean((pred_u - true_u)**2)
        loss_v: ℝ = mean((pred_v - true_v)**2)
        loss_p: ℝ = mean((pred_p - true_p)**2)
        loss: ℝ = loss_u + loss_v + loss_p
        return loss

We then minimize :math:`\mathcal{L}(\rho)` with respect to :math:`\rho` using
gradient descent via the Adam optimizer [KingmaBa2014]_ :

.. math::

    \begin{align*}
    m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
    \theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align*}

where :math:`g_t = \partial \mathcal{L}/\partial \rho` is the gradient of
the loss with respect to the density at step :math:`t`.

.. code-block:: text

    def adam(density: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        density_new: ℝ = density - lr * m_hat / (sqrt(v_hat) + eps)
        return [density_new, m_new, v_new, t + 1.0]

    ρ: ℝ = 3.0

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.01

    epochs: ℕ = 1

    for i:N(epochs):
        physika_print(i)
        g = grad(calculate_loss, ρ)
        result = adam(ρ, g, m_adam, v_adam, t_adam, lr)
        ρ = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(ρ)

At each epoch, we compute the gradient of the loss with ``grad``, update
the viscosity estimate with Adam, and print its progress toward the true
value.


Visualizing the result
^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we build a meshgrid and plot the learned velocity and pressure
fields:

.. code-block:: text

    pred_solution: ℝ[3, n_points, n_points] = solver(ρ)
    pred_u: ℝ[n_points, n_points] = pred_solution[0]
    pred_v: ℝ[n_points, n_points] = pred_solution[1]
    pred_p: ℝ[n_points, n_points] = pred_solution[2]


    X: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
    Y: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)

    for i:ℕ(n_points):
        for j:ℕ(n_points):
            X[i, j] = j * element_length
            Y[i, j] = i * element_length

    plot_navier_stokes_comparison(X, Y, true_u, true_v, true_p, pred_u, pred_v, pred_p)


.. note::
    Add ``plot_navier_stokes_comparison`` function in ``physika/runtime.py`` file:

    .. code-block:: python

        def plot_navier_stokes_comparison(X, Y, true_u, true_v, true_p, pred_u, pred_v, pred_p):
            X = X.cpu().detach().numpy()
            Y = Y.cpu().detach().numpy()
            true_u, true_v, true_p = true_u.cpu().detach().numpy(), true_v.cpu().detach().numpy(), true_p.cpu().detach().numpy()
            pred_u, pred_v, pred_p = pred_u.cpu().detach().numpy(), pred_v.cpu().detach().numpy(), pred_p.cpu().detach().numpy()

            import matplotlib.pyplot as plt
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.contourf(X[::2, ::2], Y[::2, ::2], true_p[::2, ::2], cmap="coolwarm")
            ax1.quiver(X[::2, ::2], Y[::2, ::2], true_u[::2, ::2], true_v[::2, ::2], color="black")
            ax1.streamplot(X[::2, ::2], Y[::2, ::2], true_u[::2, ::2], true_v[::2, ::2], color="black")
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_title("True")

            ax2.contourf(X[::2, ::2], Y[::2, ::2], pred_p[::2, ::2], cmap="coolwarm")
            ax2.quiver(X[::2, ::2], Y[::2, ::2], pred_u[::2, ::2], pred_v[::2, ::2], color="black")
            ax2.streamplot(X[::2, ::2], Y[::2, ::2], pred_u[::2, ::2], pred_v[::2, ::2], color="black")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title("Predicted")

            plt.tight_layout()
            plt.show()

.. figure:: /_static/tutorial_files/2d_pde/2d_navier_stokes_results.png
   :alt: Learned PDE trajectory vs ground truth
   :align: center
   :width: 700px
   :name: fig-2d-navier-stokes-results

   Figure 3: Comparison between the ground truth and learned velocity/pressure fields after training.




Full code (2D Navier stokes equation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
        return x

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results


    # --------------------------------------------------
    # Problem setup
    # --------------------------------------------------


    n_points: ℝ = 21
    domain_size: ℝ = 1.0
    n_iterations: ℝ = 500

    time_step_length: ℝ = 0.001
    ν: ℝ = 0.1
    true_ρ: ℝ = 1.0
    horizontal_velocity_top: ℝ = 1.0

    n_pressure_poisson_iterations: ℝ = 10
    stability_safety_factor: ℝ = 0.5

    element_length: ℝ = domain_size / (n_points - 1)

    x: ℝ[n_points] = linspace(0.0, domain_size, n_points)
    y: ℝ[n_points] = linspace(0.0, domain_size, n_points)



    # --------------------------------------------------
    # Discretize spatial derivatives
    # --------------------------------------------------



    def central_difference_x(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[1:n_points-1, 2:n_points] -
            f[1:n_points-1, 0:n_points-2]
        ) / (2 * element_length)
        return diff

    def central_difference_y(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[2:n_points, 1:n_points-1] -
            f[0:n_points-2, 1:n_points-1]
        ) / (2 * element_length)
        return diff

    def laplace(f: ℝ[m, n]): ℝ[m, n]:
        diff: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        diff[1:n_points-1, 1:n_points-1] = (
            f[1:n_points-1, 0:n_points-2] +   # left
            f[0:n_points-2, 1:n_points-1] +   # up
            f[1:n_points-1, 2:n_points] +     # right
            f[2:n_points, 1:n_points-1] -     # down
            4 * f[1:n_points-1, 1:n_points-1]
        ) / (element_length ** 2)
        return diff



    f: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)

    for i:ℕ(n_points):
        for j:ℕ(n_points):
            t_x = j * element_length
            t_y = i * element_length
            f[i, j] = t_x**2 + t_y**2


    # --------------------------------------------------
    # Build the solver
    # --------------------------------------------------


    n_iterations: ℝ = 5


    def solver(ρ: ℝ): ℝ[3, m, n]:
        u_prev: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        v_prev: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        p_prev: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
        for i:ℕ(n_iterations):
            d_u_prev__d_x = central_difference_x(u_prev)
            d_u_prev__d_y = central_difference_y(u_prev)
            d_v_prev__d_x = central_difference_x(v_prev)
            d_v_prev__d_y = central_difference_y(v_prev)
            laplace__u_prev = laplace(u_prev)
            laplace__v_prev = laplace(v_prev)
            u_tent = u_prev + time_step_length * (
                - (
                    u_prev * d_u_prev__d_x + v_prev * d_u_prev__d_y
                ) + ν * laplace__u_prev
            )
            v_tent = v_prev + time_step_length * (
                - (
                    u_prev * d_v_prev__d_x + v_prev * d_v_prev__d_y
                ) + ν * laplace__v_prev
            )
            u_tent[0, :] = 0.0
            u_tent[-1, :] = horizontal_velocity_top
            u_tent[:, 0] = 0.0
            u_tent[:, -1] = 0.0
            v_tent[0, :] = 0.0
            v_tent[-1, :] = 0.0
            v_tent[:, 0] = 0.0
            v_tent[:, -1] = 0.0
            d_u_tent__d_x = central_difference_x(u_tent)
            d_v_tent__d_y = central_difference_y(v_tent)
            rhs = (ρ / time_step_length * (d_u_tent__d_x + d_v_tent__d_y))
            for k:ℕ(n_pressure_poisson_iterations):
                p_next = zero_2d_array(n_points, n_points)
                p_next[1:-1, 1:-1] = 0.25 * (
                    p_prev[1:-1, :-2] +
                    p_prev[:-2, 1:-1] +
                    p_prev[1:-1, 2:] +
                    p_prev[2:, 1:-1] -
                    element_length**2 * rhs[1:-1, 1:-1]
                )
                p_next[:, -1] = p_next[:, -2]
                p_next[0, :] = p_next[1, :]
                p_next[:, 0] = p_next[:, 1]
                p_next[-1, :] = 0.0
                p_prev = p_next
            d_p_next__d_x = central_difference_x(p_next)
            d_p_next__d_y = central_difference_y(p_next)
            u_next = (
                u_tent -
                time_step_length / ρ *
                d_p_next__d_x
            )
            v_next = (
                v_tent -
                time_step_length / ρ *
                d_p_next__d_y
            )
            u_next[0, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
            u_next[-1, :] = horizontal_velocity_top
            v_next[0, :] = 0.0
            v_next[:, 0] = 0.0
            v_next[:, -1] = 0.0
            v_next[-1, :] = 0.0
            u_prev = u_next
            v_prev = v_next
            p_prev = p_next
        return [u_prev, v_prev, p_prev]





    true_solution: ℝ[3, n_points, n_points] = solver(true_ρ)
    true_u: ℝ[n_points, n_points] = true_solution[0]
    true_v: ℝ[n_points, n_points] = true_solution[1]
    true_p: ℝ[n_points, n_points] = true_solution[2]



    # --------------------------------------------------
    # Define loss function and optimizer
    # --------------------------------------------------


    def calculate_loss(ρ: ℝ): ℝ:
        predictions: ℝ[3, n_points, n_points] = solver(ρ)
        pred_u: ℝ[n_points, n_points] = predictions[0]
        pred_v: ℝ[n_points, n_points] = predictions[1]
        pred_p: ℝ[n_points, n_points] = predictions[2]
        loss_u: ℝ = mean((pred_u - true_u)**2)
        loss_v: ℝ = mean((pred_v - true_v)**2)
        loss_p: ℝ = mean((pred_p - true_p)**2)
        loss: ℝ = loss_u + loss_v + loss_p
        return loss

    def adam(ρ: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        ρ_new: ℝ = ρ - lr * m_hat / (sqrt(v_hat) + eps)
        return [ρ_new, m_new, v_new, t + 1.0]



    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------


    ρ: ℝ = 3.0
    #guess_solution = solver(ρ)

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.01

    epochs: ℕ = 400

    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, ρ)
        result = adam(ρ, g, m_adam, v_adam, t_adam, lr)
        ρ = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]
        physika_print(ρ)

    # value of `ρ` should be close to 1.0
    ρ

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------


    pred_solution: ℝ[3, n_points, n_points] = solver(ρ)
    pred_u: ℝ[n_points, n_points] = pred_solution[0]
    pred_v: ℝ[n_points, n_points] = pred_solution[1]
    pred_p: ℝ[n_points, n_points] = pred_solution[2]


    X: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)
    Y: ℝ[n_points, n_points] = zero_2d_array(n_points, n_points)

    for i:ℕ(n_points):
        for j:ℕ(n_points):
            X[i, j] = j * element_length
            Y[i, j] = i * element_length


    plot_navier_stokes_comparison(X, Y, true_u, true_v, true_p, pred_u, pred_v, pred_p)



References
----------

.. [NavierStokesWiki] Wikipedia contributors. *Navier–Stokes Equations*.
   https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations

.. [ANLINSChorin] Argonne National Laboratory. *INSChorin Module*.
   Cardinal Documentation.
   https://cardinal.cels.anl.gov/modules/navier_stokes/inschorin.html

.. [CadenceHeat2D] Cadence System Analysis. *Using the 2D Finite Difference
   Method for Heat Transfer Analysis*.
   https://resources.system-analysis.cadence.com/blog/msa2022-using-the-2d-finite-difference-method-for-heat-transfer-analysis

.. [SimScaleCFL] SimScale. *What Is the CFL Condition?*
   https://www.simscale.com/blog/cfl-condition/

.. [KingmaBa2014] Kingma, D. P., & Ba, J. (2014).
   *Adam: A Method for Stochastic Optimization*.
   https://arxiv.org/pdf/1412.6980

.. [AdamsWave2D] Adams, V. H.
   *Finite Difference Discretization of the 2D Wave Equation*.
   https://vanhunteradams.com/DE1/Drum/Discretization.html

.. [CadenceNavierStokes] Cadence System Analysis.
   *Formulating the 2D Incompressible Steady-State Navier–Stokes Equation*.
   https://resources.system-analysis.cadence.com/blog/msa2022-formulating-the-2d-incompressible-steady-state-navier-stokes-equation


- Navier-Stokes equation (lid-driven cavity):
  `Ceyron, "lid_driven_cavity_python_simple.py", machine-learning-and-simulation <https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lid_driven_cavity_python_simple.py>`_

- Navier-Stokes equation (numerical background):
  `Matyka, M., "Solution to two-dimensional Incompressible Navier-Stokes Equations with SIMPLE, SIMPLER and Vorticity-Stream Function Approaches. Driven-Lid Cavity Problem: Solution and Visualization" <https://arxiv.org/pdf/physics/0407002>`_
