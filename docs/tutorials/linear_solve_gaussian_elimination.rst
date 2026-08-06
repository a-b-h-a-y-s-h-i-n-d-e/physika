Linear solve using Gaussian Elimination
============================================

In this tutorial we will learn how to solve linear equations using Linear solve method, particularly
Gaussian Elimination method [Wikipedia_GaussianElim]_ .

The Equation
------------

Following is the linear equations which we are going to solve:

.. math::
 
   \begin{aligned}
   x + 2y + z &= 8 \\
   3x + y - z &= 2 \\
   2x - y + z &= 3
   \end{aligned}

First we will write this equations in form of ``Ax = B`` which is as follow:

.. math::

    \underset{A}{
    \begin{bmatrix}
    1 & 2 & 1 \\
    3 & 1 & -1 \\
    2 & -1 & 1
    \end{bmatrix}}
    \underset{x}{
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}}
    =
    \underset{B}{
    \begin{bmatrix}
    8 \\
    2 \\
    3
    \end{bmatrix}}


In physika we define this matrices such as:

.. code-block:: text

    A: ℝ[3,3] = [
        [1, 2, 1],
        [3, 1, -1],
        [2, -1, 1]
    ]
    b: ℝ[3] = [8, 2, 3]



Gaussian elimination method
------------------------------

This section will get divided into 3 subsections:

* Build the augmented matrix
* Perform Forward elimination
* Back substitution to find values

Before starting lets create a function as:

.. code-block:: text

    def gaussian_solve(A: ℝ[m, n], b: ℝ[n]): ℝ[m]:
        ...

- ``A: ℝ[m, n]`` - represents ``A`` matrix
- ``b: ℝ[n]`` - represents ``b`` matrix
- ``ℝ[m]`` - represents return type, which will be solution vector for x, y, z values
  
Step 1 - Augmented matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step we will merge matrix :math:`A` and :math:`b` in single matrix called as Augmented matrix, so that 
every row operations (follow up steps) gets applied to both the matrices at the same time.


.. code-block:: text

    a_row: ℝ = get_2d_array_num_rows(A)
    a_col: ℝ = get_2d_array_num_cols(A)

    new_col: ℝ = a_col + 1
    aug: ℝ[a_row, new_col] = zeros(a_row, new_col)
    for i:ℕ(a_row):
        aug[i, :a_col] = A[i, :]
        aug[i, a_col] = b[i]

Since the :math:`A` matrix is of size ``3x3`` and :math:`b` is of size ``1x3``, the augmented matrix will have shape of ``3x4``
so we loop through number of rows of :math:`A` which is 3, and add each row from :math:`A` ``aug[i, :a_col] = A[i, :]``
and :math:`b` ``aug[i, a_col] = b[i]`` together into the ``aug`` matrix row.

After that augmented matrix looks like this:

.. math::

    \left[\begin{array}{ccc|c}
    1 &  2 &  1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 &  1 & 3
    \end{array}\right]


Step 2 - Forward elimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section will get further divied into 3 sub-section, which are

- Partial pivoting
- swap rows using buffer
- Elimination

We will go through each of this in detail, but to give some context, here is the physika code:


.. code-block:: text

    # -------------------------
    # Forward elimination
    # -------------------------
    row_buffer: ℝ[new_col] = zeros(new_col)
    for i:ℕ(a_row):
        # -------------------------
        # Partial pivoting
        # -------------------------
        max_row = i
        for k:ℕ(i + 1, a_row):
            if abs(aug[k, i]) > abs(aug[max_row, i]):
                max_row = k
        # -------------------------
        # Swap rows using buffer
        # -------------------------
        if max_row != i:
            for k:ℕ(new_col):
                row_buffer[k] = aug[i, k]
            for k:ℕ(new_col):
                aug[i, k] = aug[max_row, k]
            for k:ℕ(new_col):
                aug[max_row, k] = row_buffer[k]
        # -------------------------
        # Elimination
        # -------------------------
        for j:ℕ(i + 1, a_row):
            factor = aug[j, i] / aug[i, i]
            for k:ℕ(i, new_col):
                aug[j, k] = aug[j, k] - factor * aug[i, k]


``row_buffer`` is temporary vector which will be used to swap rows in section: 2.2, this is same concept as swap values two variables using third variable.
``new_col`` represents length of columns of augmented matrix, which is 4.
The outer for loop will loop through each row of the augmented matrix ``a_row`` which value is 3.

Now we will go step by step in first iteration of outer loop.

2.1 Partial pivoting
********************

.. code-block:: text

    # -------------------------
    # Partial pivoting
    # -------------------------
    max_row = i
    for k:ℕ(i + 1, a_row):
        if abs(aug[k, i]) > abs(aug[max_row, i]):
            max_row = k

The pivot values are the diagonal values so for the first iteration the pivot is
at first value of first row, column which is denoted by red box in below matrix [Chasnov_PartialPivot]_ 

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\boxed{1}} & 2 & 1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 & 1 & 3
    \end{array}\right]


The ``k`` loop here starts from row number 1, and pluck out each number from first column:

.. math::

    \begin{bmatrix}
    \color{red}{\boxed{1}} \\
    3 \\
    2
    \end{bmatrix}

and then sequentially (column wise) we compare each values with each other and update the ``max_row`` value.
For example:

.. math::

   \begin{array}{cccc}
   \text{Step} & \text{Comparison} & \text{Decision} & max\_row \\[1.5ex]
   \hline \\[-1ex]
   \text{Init} & - & - & \mathbf{0} \\[1.5ex]
   k = 1 & |3| > |1| & \color{green}{\checkmark \text{ Update}} & \mathbf{1} \\[1.5ex]
   k = 2 & |2| > |3| & \color{gray}{\times \text{ Skip}} & \mathbf{1}
   \end{array}

Now after this value of ``max_row`` gets updated to 1 which is second row.


2.2 swap rows using buffer
***************************


.. code-block:: text

    # -------------------------
    # Swap rows using buffer
    # -------------------------
    if max_row != i:
        for k:ℕ(new_col):
            row_buffer[k] = aug[i, k]
        for k:ℕ(new_col):
            aug[i, k] = aug[max_row, k]
        for k:ℕ(new_col):
            aug[max_row, k] = row_buffer[k]

The ``i`` value is 0 which is value of first row, and the ``max_row`` just got updated as 1 which is second row, It means
this if-block will get executed and the first row will get swap with second row



Before row swap:

.. math::

    \left[\begin{array}{ccc|c}
    1 & 2 & 1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 & 1 & 3
    \end{array}\right]
    \begin{array}{l}
    \left.\begin{array}{c} ~ \\ ~ \end{array}\right\} \text{Swap} \\
    ~
    \end{array}

After row swap:

.. math::

    \left[\begin{array}{ccc|c}
    \color{green}{\mathbf{3}} & \color{green}{\mathbf{1}} & \color{green}{\mathbf{-1}} & \color{green}{\mathbf{2}} \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]


also after swapping the pivot value also gets updated now which is 3:


.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\boxed{3}} & 1 & -1 & 2 \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]

2.3 Elimination
***************************

.. code-block:: text

    # -------------------------
    # Elimination
    # -------------------------
    for j:ℕ(i + 1, a_row):
        factor = aug[j, i] / aug[i, i]
        for k:ℕ(i, new_col):
            aug[j, k] = aug[j, k] - factor * aug[i, k]
        

Once the row-swapping is done, we move to Elimination section where we transform our augmented matrix into upper-triangular matrix
so for this first iteration we will eliminate all the entries below the pivot value to zeros.

To do that, we start the ``j`` loop from second row, since our first iteration starts with first row (outer loop)
and we make a ``factor`` value:

.. math::

   \text{factor} = \frac{\text{Target Element}}{\text{Pivot Element}} = \frac{\text{aug}[j, i]}{\text{aug}[i, i]}


Then, we update the target row :math:`j` using row operations:

.. math::

   \text{Row}_j \leftarrow \text{Row}_j - (\text{factor} \times \text{Row}_i)


From our previous row-swapping step, our updated matrix is:

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\mathbf{3}} & 1 & -1 & 2 \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]

Here, the pivot element is :math:`\text{aug}[0, 0] = \color{red}{3}`. The outer loop ``j`` iterates through rows below row 0, namely **Row 1** (:math:`j=1`) and **Row 2** (:math:`j=2`).


Second row operation
^^^^^^^^^^^^^^^^^^^^

1. Calculate Factor:
   
   .. math::

       \text{factor} = \frac{\text{aug}[1, 0]}{\text{aug}[0, 0]} = \frac{1}{3}

2. Apply Row Operation: :math:`\text{Row}_2 \leftarrow \text{Row}_2 - \frac{1}{3} \times \text{Row}_1`

   .. math::

      \begin{array}{rcccl}
      \text{Row}_2 \text{ (Original):} & [1, & 2, & 1, & 8] \\
      - \left(\frac{1}{3} \times \text{Row}_1\right): & -\left[1, \right. & \frac{1}{3}, & -\frac{1}{3}, & \left. \frac{2}{3}\right] \\[1ex]
      \hline \\[-1.5ex]
      \text{New Row}_2: & [\mathbf{0}, & \mathbf{\frac{5}{3}}, & \mathbf{\frac{4}{3}}, & \mathbf{\frac{22}{3}}]
      \end{array}

Augmented matrix after second row operation:

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\mathbf{3}} & 1 & -1 & 2 \\
    \color{green}{\mathbf{0}} & \frac{5}{3} & \frac{4}{3} & \frac{22}{3} \\
    2 & -1 & 1 & 3
    \end{array}\right]


Third row operation
^^^^^^^^^^^^^^^^^^^^

1. Calculate Factor:

   .. math::

       \text{factor} = \frac{\text{aug}[2, 0]}{\text{aug}[0, 0]} = \frac{2}{3}

2. Apply Row Operation: :math:`\text{Row}_3 \leftarrow \text{Row}_3 - \frac{2}{3} \times \text{Row}_1`

   .. math::

      \begin{array}{rcccl}
      \text{Row}_3 \text{ (Original):} & [2, & -1, & 1, & 3] \\
      - \left(\frac{2}{3} \times \text{Row}_1\right): & -\left[2, \right. & \frac{2}{3}, & -\frac{2}{3}, & \left. \frac{4}{3}\right] \\[1ex]
      \hline \\[-1.5ex]
      \text{New Row}_3: & [\mathbf{0}, & \mathbf{-\frac{5}{3}}, & \mathbf{\frac{5}{3}}, & \mathbf{\frac{5}{3}}]
      \end{array}


Now after third and last row operation, the augmented matrix will look like this:


.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\mathbf{3}} & 1 & -1 & 2 \\
    \color{green}{\mathbf{0}} & \frac{5}{3} & \frac{4}{3} & \frac{22}{3} \\
    \color{green}{\mathbf{0}} & -\frac{5}{3} & \frac{5}{3} & \frac{5}{3}
    \end{array}\right]



This completes the first iteration of the outer loop ``(i = 1)``. However, our goal is to transform the
augmented matrix into upper-triangular matrix form.
Therefore, the next outer loop will run second iteration ``(i = 2)``, which will also repeat the three core
steps: finding the pivot, swapping rows, and performing elimination.
and after all this, our augmented matrix will get transformed into upper-triangular such as:


.. math::

    \left[\begin{array}{ccc|c}
    \mathbf{3} & -1 & -1 & 2 \\[1ex]
    0 & \mathbf{\frac{5}{3}} & \frac{4}{3} & \frac{22}{3} \\[1ex]
    0 & 0 & \mathbf{3} & 9
    \end{array}\right]



Step 3 - Back substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Once the forward elimination transforms augmented matrix into upper-triangular matrix, we use Back substitution to find values
of x, y, and z.
To understand why this is called as "Back substitution", lets take a look at our final upper-triangular matrix

.. math::

    \left[\begin{array}{ccc|c}
    \mathbf{3} & 1 & -1 & 2 \\[1ex]
    0 & \mathbf{\frac{5}{3}} & \frac{4}{3} & \frac{22}{3} \\[1ex]
    0 & 0 & \mathbf{3} & 9
    \end{array}\right]

now lets convert each row into system of linear equations, just like at beginning of the tutorial we converted linear equations into matrix form,
here we convert matrix form into linear equations:


.. math::

   \begin{aligned}
   3x + y - z &= 2 \\[1ex]
   \frac{5}{3}y + \frac{4}{3}z &= \frac{22}{3} \\[1ex]
   3z &= 9
   \end{aligned}

Notice how the third equation contains only one variable :math:`z` which we can easily solve and find value of :math:`z`:


Solve for :math:`z` (Row 3)
****************************************

From the third equation:

.. math::

   3z = 9

.. math::

   z = \frac{9}{3} = \mathbf{3}

Solve for :math:`y` (Row 2)
****************************************

Substitute :math:`z = 3` into the second equation:

.. math::

   \frac53y + \frac43(3) = \frac{22}3

.. math::

   \frac53y + 4 = \frac{22}3

.. math::

   \frac53y = \frac{22}3 - \frac{12}3 = \frac{10}3

.. math::

   y = \frac{10}3 \times \frac35 = \mathbf2


Solve for :math:`x` (Row 1)
****************************************

Substitute :math:`y = 2` and :math:`z = 3` into the first equation:

.. math::

   3x + 2 - 3 = 2

.. math::

   3x - 1 = 2

.. math::

   3x = 3

.. math::

   x = \frac33 = \mathbf1


Therefore, the final solution vector is

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z
    \end{bmatrix}
    =
    \begin{bmatrix}
    1\\
    2\\
    3
    \end{bmatrix}

We can do this in Physika code by using below code:

.. code-block:: text

    # -------------------------
    # Back substitution
    # -------------------------
    x: ℝ[a_col] = zeros(a_col)
    for i:ℕ(a_col):
        idx = a_col - 1 - i
        total = aug[idx, a_col]
        for j:ℕ(idx + 1, a_row):
            total = total - aug[idx, j] * x[j]
        x[idx] = total / aug[idx, idx]



Full code
---------

.. code-block:: text

    # ---------------------
    # Helper functions
    # ---------------------

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_rows(x: ℝ[m, n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total


    def get_2d_array_num_cols(x: ℝ[m, n]): ℝ:
        return get_1d_array_length(x[0])

    def arange(n: ℕ): ℝ[n]:
        arr: ℝ[n] = for i: ℕ(n) → i
        return arr



    def gaussian_solve(A: ℝ[m, n], b: ℝ[n]): ℝ[m]:
        a_row: ℝ = get_2d_array_num_rows(A)
        a_col: ℝ = get_2d_array_num_cols(A)
        # -------------------------
        # Create augmented matrix
        # -------------------------
        new_col: ℝ = a_col + 1
        aug: ℝ[a_row, new_col] = zeros(a_row, new_col)
        for i:ℕ(a_row):
            aug[i, :a_col] = A[i, :]
            aug[i, a_col] = b[i]
        # -------------------------
        # Forward elimination
        # -------------------------
        row_buffer: ℝ[new_col] = zeros(new_col)
        for i:ℕ(a_row):
            # -------------------------
            # Partial pivoting
            # -------------------------
            max_row = i
            for k:ℕ(i + 1, a_row):
                if abs(aug[k, i]) > abs(aug[max_row, i]):
                    max_row = k
            # -------------------------
            # Swap rows using buffer
            # -------------------------
            if max_row != i:
                for k:ℕ(new_col):
                    row_buffer[k] = aug[i, k]
                for k:ℕ(new_col):
                    aug[i, k] = aug[max_row, k]
                for k:ℕ(new_col):
                    aug[max_row, k] = row_buffer[k]
            # -------------------------
            # Elimination
            # -------------------------
            for j:ℕ(i + 1, a_row):
                factor = aug[j, i] / aug[i, i]
                for k:ℕ(i, new_col):
                    aug[j, k] = aug[j, k] - factor * aug[i, k]
        # -------------------------
        # Back substitution
        # -------------------------
        x: ℝ[a_col] = zeros(a_col)
        for i:ℕ(a_col):
            idx = a_col - 1 - i
            total = aug[idx, a_col]
            for j:ℕ(idx + 1, a_row):
                total = total - aug[idx, j] * x[j]
            x[idx] = total / aug[idx, idx]
        return x



    A: ℝ[3,3] = [
        [1, 2, 1],
        [3, 1, -1],
        [2, -1, 1]
    ]
    b: ℝ[3] = [8, 2, 3]

    gaussian_solve(A, b)



References
----------

.. [Wikipedia_GaussianElim] Wikipedia contributors, *Gaussian elimination*, Wikipedia,
  The Free Encyclopedia. https://en.wikipedia.org/wiki/Gaussian_elimination
.. [Chasnov_PartialPivot] J. R. Chasnov, *Partial Pivoting*, in Numerical Methods,
  LibreTexts, Hong Kong University of Science and Technology.
  https://math.libretexts.org/Bookshelves/Applied_Mathematics/Numerical_Methods_(Chasnov)/03%3A_System_of_Equations/3.03%3A_Partial_Pivoting