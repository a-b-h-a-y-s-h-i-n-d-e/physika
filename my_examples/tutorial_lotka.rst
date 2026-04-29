Parameter Learning for ODEs: Lotka-Volterra
============================================

In this tutorial we will learn the parameters of the Lotka-Volterra equations
— a pair of coupled ODEs that model predator-prey dynamics in ecology. We will
use the adjoint method for efficient gradient computation.

The Equations
-------------

The Lotka-Volterra system is:

.. math::

   \frac{dx}{dt} = \alpha x - \beta xy

   \frac{dy}{dt} = \delta xy - \gamma y

Where :math:`x` is the prey population, :math:`y` is the predator population,
and :math:`\theta = [\alpha, \beta, \gamma, \delta]` are the parameters we want
to learn.


Helper functions
----------------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = len(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        for i:ℕ(new_length):
            if i<len(x):
                results[i] = x[i]
            else:
                results[i] = var
        return results


Step 1: Define the ODE
-----------------------

.. code-block:: text

    def f(state: ℝ[2], θ: ℝ[4]): ℝ:
        x: ℝ = state[0]
        y: ℝ = state[1]
        α: ℝ = θ[0]
        β: ℝ = θ[1]
        γ: ℝ = θ[2]
        δ: ℝ = θ[3]
        dx: ℝ = (α * x) - (β * x * y)
        dy: ℝ = - (γ * y) + (δ * x * y)
        return [dx, dy]

``f`` takes the current state ``[x, y]`` and parameters ``theta`` and returns
the derivatives ``[dx/dt, dy/dt]``.


Step 2: Build the RK4 Solver
-----------------------------

We use the Runge-Kutta 4 method for higher accuracy than Forward Euler.
RK4 evaluates the derivative at four points within each time step:

.. code-block:: text

    def rk4_step(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = [state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]]
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = [state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]]
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = [state[0] + dt * k3[0], state[1] + dt * k3[1]]
        k4: ℝ[2] = f(k4_state, θ)
        x_new: ℝ = state[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y_new: ℝ = state[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        return [x_new, y_new]
    

Step 3: Build the Trajectory Solver
-------------------------------------

We integrate the system forward from initial conditions
:math:`x_0 = 10, y_0 = 1` over 100 timesteps:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 100

    def solver(θ: ℝ[4]): ℝ[m]:
        state: ℝ[2] = [10.0, 1.0]
        x_array: ℝ[0] = [10.0]
        y_array: ℝ[0] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            x = results[0]
            y = results[1]
            x_array = append(x_array, x)
            y_array = append(y_array, y)
            state = results
        return [x_array, y_array]


Step 4: Generate Ground Truth Data
------------------------------------

We pick true parameters and generate the ground truth trajectories:

.. code-block:: text

    true_theta: ℝ[4] = [1.5, 1.0, 3.0, 1.0]
    true_results: ℝ[m] = solver(true_theta)
    true_x: ℝ[m] = true_results[0]
    true_y: ℝ[m] = true_results[1]


Step 5: Adjoint Gradient
------------------------

To compute the gradients we use the adjoint method, which gives
better convergence for ODE parameter estimation:

.. code-block:: text

    def adjoint_grad(θ: ℝ[4]): ℝ:
        states: ℝ[m] = solver(θ)
        x_array: ℝ[m] = states[0]
        y_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(x_array)
        s: ℝ[2] = [
            2 * (x_array[m-1] - true_x[m-1]) / m,
            2 * (y_array[m-1] - true_y[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 1 - i
            x = x_array[idx]
            y = y_array[idx]
            state = [x, y]
            J_state = grad(rk4_step, state, θ, 0)
            J_theta = grad(rk4_step, state, θ, 1)
            L += s @ J_theta
            s = s @ J_state
        return L

``grad(rk4_step, state, theta, 0)`` differentiates ``rk4_step`` with respect
to ``state`` (argument 0) and ``grad(rk4_step, state, theta, 1)`` with respect
to ``theta`` (argument 1).

Step 6: Train with Gradient Descent
-----------------------------------

We start with an initial guess and run 1000 epochs:

.. code-block:: text

    θ: ℝ[4] = [1.0, 0.7, 2.5, 0.7]  
    learning_rate: ℝ = 0.05
    epochs: ℕ = 1000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

After training, ``theta`` should be close to ``[1.5, 1.0, 3.0, 1.0]``.

Step 7: Visualize Results
--------------------------

.. code-block:: text

   pred_results = solver(theta)
   plot_trajectories(true_results, pred_results)

.. note::
   ``plot_trajectories`` is not a built-in Physika function. Add the
   following helper function to ``physika/runtime.py``:

   .. code-block:: python

        def plot_trajectories(true_results, pred_results):
            import matplotlib.pyplot as plt

            plt.plot(true_results[0, :], color="orange", label="Prey (True)")
            plt.plot(pred_results[0, :], color="orange", linestyle="--", label="Prey (Predicted)")

            plt.plot(true_results[1, :], color="green", label="Predator (True)")
            plt.plot(pred_results[1, :], color="green", linestyle="--", label="Predator (Predicted)")

            plt.xlabel("Time Step")
            plt.ylabel("Population")
            plt.title("Lotka-Volterra: True vs Predicted")
            plt.legend()
            plt.show()

.. figure:: /_static/tutorial_files/__.png
   :alt: Predicted trajecotory vs ground truth
   :align: center
   :width: 700px

   Comparison between ground truth and learned trajectory after training.

Full Code
---------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = len(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        for i:ℕ(new_length):
            if i<len(x):
                results[i] = x[i]
            else:
                results[i] = var
        return results

    def f(state: ℝ[2], θ: ℝ[4]): ℝ:
        x: ℝ = state[0]
        y: ℝ = state[1]
        α: ℝ = θ[0]
        β: ℝ = θ[1]
        γ: ℝ = θ[2]
        δ: ℝ = θ[3]
        dx: ℝ = (α * x) - (β * x * y)
        dy: ℝ = - (γ * y) + (δ * x * y)
        return [dx, dy]

    def rk4_step(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = [state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]]
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = [state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]]
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = [state[0] + dt * k3[0], state[1] + dt * k3[1]]
        k4: ℝ[2] = f(k4_state, θ)
        x_new: ℝ = state[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y_new: ℝ = state[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        return [x_new, y_new]
    
    dt: ℝ = 0.1
    timesteps: ℝ = 100

    def solver(θ: ℝ[4]): ℝ[m]:
        state: ℝ[2] = [10.0, 1.0]
        x_array: ℝ[0] = [10.0]
        y_array: ℝ[0] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            x = results[0]
            y = results[1]
            x_array = append(x_array, x)
            y_array = append(y_array, y)
            state = results
        return [x_array, y_array]
    
    true_theta: ℝ[4] = [1.5, 1.0, 3.0, 1.0]
    true_results: ℝ[m] = solver(true_theta)
    true_x: ℝ[m] = true_results[0]
    true_y: ℝ[m] = true_results[1]

    def adjoint_grad(θ: ℝ[4]): ℝ:
        states: ℝ[m] = solver(θ)
        x_array: ℝ[m] = states[0]
        y_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(x_array)
        s: ℝ[2] = [
            2 * (x_array[m-1] - true_x[m-1]) / m,
            2 * (y_array[m-1] - true_y[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 1 - i
            x = x_array[idx]
            y = y_array[idx]
            state = [x, y]
            J_state = grad(rk4_step, state, θ, 0)
            J_theta = grad(rk4_step, state, θ, 1)
            L += s @ J_theta
            s = s @ J_state
        return L

    θ: ℝ[4] = [1.0, 0.7, 2.5, 0.7]  
    learning_rate: ℝ = 0.05
    epochs: ℕ = 1000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g
    
    pred_results = solver(theta)
    plot_trajectories(true_results, pred_results)


References
----------

- `Lotka-Volterra equations — Wikipedia <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
- `Adjoint Differentiation — MIT OCW <https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/ocw_18s096_lecture06-part1_2023jan30_mp4/>`_
