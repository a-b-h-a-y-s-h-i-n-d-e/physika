import re


def from_ast_to_torch(unified_ast, print_code=True):
    """
    Convert unified AST to PyTorch code and evaluate it.

    Args:
        unified_ast: The unified AST from build_unified_ast()
        print_code: If True, print the generated PyTorch code

    Returns:
        List of evaluation results
    """
    code_lines = []
    results = []

    # Track generated variable names
    var_counter = [0]

    def fresh_var():
        var_counter[0] += 1
        return f"_t{var_counter[0]}"

    def type_to_torch(type_spec):
        """Convert physika type to PyTorch type annotation."""
        if isinstance(type_spec, str):
            if type_spec == "ℝ":
                return "torch.Tensor"
            elif type_spec == "ℕ":
                return "int"
            return type_spec
        elif isinstance(type_spec, tuple):
            if type_spec[0] == "tensor":
                dims = type_spec[1]
                shape = [d[0] for d in dims]
                return f"torch.Tensor  # shape {shape}"
            elif type_spec[0] == "func_type":
                return f"Callable[{type_spec[1]}, {type_spec[2]}]"
        return str(type_spec)

    # Track current loop variable for imaginary -> loop var conversion
    current_loop_var = [None]

    def ast_to_torch_expr(node, indent=0):
        """Convert an AST expression node to PyTorch code string."""
        if not isinstance(node, tuple):
            return repr(node)

        op = node[0]

        if op == "num":
            val = node[1]
            if isinstance(val, float) and val == int(val):
                return f"{val}"
            return repr(val)

        elif op == "var":
            return node[1]

        elif op == "add":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} + {right})"

        elif op == "sub":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} - {right})"

        elif op == "mul":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} * {right})"

        elif op == "div":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} / {right})"

        elif op == "matmul":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} @ {right})"

        elif op == "pow":
            left = ast_to_torch_expr(node[1], indent)
            right = ast_to_torch_expr(node[2], indent)
            return f"({left} ** {right})"

        elif op == "neg":
            val = ast_to_torch_expr(node[1], indent)
            return f"(-{val})"

        elif op == "array":
            elements = node[1]
            # Check if this is a nested array (contains other arrays)
            has_nested = any(isinstance(e, tuple) and e[0] == "array" for e in elements)
            if has_nested:
                # For nested arrays, generate list-of-lists and wrap in torch.tensor
                def array_to_list(arr_node):
                    if isinstance(arr_node, tuple) and arr_node[0] == "array":
                        inner = [array_to_list(e) for e in arr_node[1]]
                        return f"[{', '.join(inner)}]"
                    else:
                        return ast_to_torch_expr(arr_node, indent)
                inner_lists = [array_to_list(e) for e in elements]
                return f"torch.tensor([{', '.join(inner_lists)}])"
            else:
                all_numeric = all(isinstance(e, tuple) and e[0] == "num" for e in elements)
                elem_strs = [ast_to_torch_expr(e, indent) for e in elements]
                if all_numeric:
                    return f"torch.tensor([{', '.join(elem_strs)}])"
                else:
                    # Elements may be tensors (e.g., x[1], sin(x[0])) — use torch.stack
                    wrapped = [f"torch.as_tensor({s}).float()" for s in elem_strs]
                    return f"torch.stack([{', '.join(wrapped)}])"

        elif op == "index":
            var_name = node[1]
            idx = ast_to_torch_expr(node[2], indent)
            return f"{var_name}[int({idx})]"

        elif op == "slice":
            var_name = node[1]
            start = ast_to_torch_expr(node[2], indent)
            end = ast_to_torch_expr(node[3], indent)
            # Convert to int if needed
            start_int = f"int({start})" if "." in start else start
            end_int = f"int({end})+1" if "." in end else f"{end}+1"
            return f"{var_name}[{start_int}:{end_int}]"

        elif op == "call":
            func_name = node[1]
            args = node[2]
            arg_strs = [ast_to_torch_expr(arg, indent) for arg in args]

            # Map built-in functions to PyTorch equivalents
            torch_funcs = {
                "exp": "torch.exp",
                "log": "torch.log",
                "sin": "torch.sin",
                "cos": "torch.cos",
                "sqrt": "torch.sqrt",
                "abs": "torch.abs",
                "sum": "torch.sum",
                "mean": "torch.mean",
                "real": "torch.real",
            }

            if func_name in torch_funcs:
                return f"{torch_funcs[func_name]}({', '.join(arg_strs)})"
            elif func_name == "grad":
                # grad(output, input) -> compute_grad(output, input)
                return f"compute_grad({', '.join(arg_strs)})"
            else:
                return f"{func_name}({', '.join(arg_strs)})"

        elif op == "call_index":
            # Indexed function call: func(args)[index]
            func_name = node[1]
            args = node[2]
            index_ast = node[3]
            arg_strs = [ast_to_torch_expr(arg, indent) for arg in args]
            idx = ast_to_torch_expr(index_ast, indent)

            if func_name == "grad":
                # grad(output, input)[i] -> compute_grad(output, input)[i]
                return f"compute_grad({', '.join(arg_strs)})[int({idx})]"
            else:
                return f"{func_name}({', '.join(arg_strs)})[int({idx})]"

        elif op == "imaginary":
            # If we're inside a for loop with loop var 'i', use the loop var
            if current_loop_var[0] == "i":
                return "i"
            # Use torch.tensor(1j) so it can be used with torch.exp
            return "torch.tensor(1j)"

        elif op == "equation_string":
            return repr(node[1])

        elif op == "string":
            # Equation string literal
            return repr(node[1])

        return f"/* unknown: {node} */"

    def generate_function(name, func_def):
        """Generate PyTorch code for a function definition."""
        params = func_def["params"]
        body = func_def["body"]
        statements = func_def.get("statements", [])

        # Build parameter list
        param_strs = []
        param_names = []
        for param_name, param_type in params:
            type_str = type_to_torch(param_type)
            param_strs.append(f"{param_name}")
            param_names.append(param_name)

        lines = [f"def {name}({', '.join(param_strs)}):"]

        # Track known variables (params + locals)
        known_vars = list(param_names)

        # Track equation string variable names
        equation_vars = set()

        # Helper to generate solve call with known variables
        def generate_solve_call(expr):
            if isinstance(expr, tuple) and expr[0] == "call" and expr[1] == "solve":
                args = expr[2]
                arg_strs = [ast_to_torch_expr(arg) for arg in args]
                # Add known variables as keyword arguments (exclude equation vars)
                kw_strs = [f"{v}={v}" for v in known_vars if v not in equation_vars]
                return f"solve({', '.join(arg_strs)}, {', '.join(kw_strs)})"
            return ast_to_torch_expr(expr)

        # Generate body statements
        for stmt in statements:
            if stmt is None:
                continue
            stmt_op = stmt[0]
            if stmt_op == "body_decl":
                _, var_name, var_type, expr = stmt
                # Track if this is an equation string
                if isinstance(expr, tuple) and expr[0] == "string":
                    equation_vars.add(var_name)
                expr_code = generate_solve_call(expr)
                lines.append(f"    {var_name} = {expr_code}")
                known_vars.append(var_name)
            elif stmt_op == "body_assign":
                _, var_name, expr = stmt
                expr_code = generate_solve_call(expr)
                lines.append(f"    {var_name} = {expr_code}")
                known_vars.append(var_name)
            elif stmt_op == "body_tuple_unpack":
                _, var_names, expr = stmt
                expr_code = generate_solve_call(expr)
                lines.append(f"    {', '.join(var_names)} = {expr_code}")
                known_vars.extend(var_names)

        # Generate return statement
        body_code = ast_to_torch_expr(body)
        lines.append(f"    return {body_code}")

        return "\n".join(lines)

    def generate_class(name, class_def):
        """Generate PyTorch code for a class definition."""
        class_params = class_def["class_params"]
        lambda_params = class_def["lambda_params"]
        body = class_def["body"]
        has_loop = class_def.get("has_loop", False)
        loop_var = class_def.get("loop_var")
        loop_body = class_def.get("loop_body", [])
        has_loss = class_def.get("has_loss", False)
        loss_body = class_def.get("loss_body")

        lines = [f"class {name}(nn.Module):"]

        # __init__ method
        init_params = ", ".join([p[0] for p in class_params])
        lines.append(f"    def __init__(self, {init_params}):")
        lines.append(f"        super().__init__()")
        for param_name, param_type in class_params:
            # Check if this is a tensor type that should be a parameter
            is_tensor = False
            if isinstance(param_type, tuple) and param_type[0] == "tensor":
                is_tensor = True
            elif param_type == "ℝ":
                is_tensor = True  # Scalar could be a learnable parameter

            if is_tensor:
                # Handle both tensors and scalars
                lines.append(f"        self.{param_name} = nn.Parameter(torch.tensor({param_name}).float() if not isinstance({param_name}, torch.Tensor) else {param_name}.clone().detach().float())")
            else:
                # Non-tensor (like function 'f' or int 'n')
                lines.append(f"        self.{param_name} = {param_name}")

        # forward method (lambda)
        lambda_param_names = [p[0] for p in lambda_params]
        lines.append(f"")
        lines.append(f"    def forward(self, {', '.join(lambda_param_names)}):")
        # Convert inputs to tensors
        for pname, ptype in lambda_params:
            if ptype == "ℝ" or ptype == "ℕ" or (isinstance(ptype, tuple) and ptype[0] == "tensor"):
                lines.append(f"        {pname} = torch.as_tensor({pname}).float()")

        # Helper to replace class params with self.param in expressions
        def replace_class_params(code, class_params):
            for cp_name, _ in class_params:
                # Replace function calls: f(...) -> self.f(...)
                code = re.sub(rf'\b{cp_name}\(', f'self.{cp_name}(', code)
                # Replace array indexing: W[...] -> self.W[...]
                code = re.sub(rf'\b{cp_name}\[', f'self.{cp_name}[', code)
                # Replace standalone references in expressions
                code = re.sub(rf'\(({cp_name})\s', r'(self.\1 ', code)
                code = re.sub(rf'\s({cp_name})\)', r' self.\1)', code)
                code = re.sub(rf'\(({cp_name})\)', r'(self.\1)', code)
            return code

        # Generate loop if present
        if has_loop and loop_body:
            lines.append(f"        n = int(self.n) if hasattr(self, 'n') else self.{class_params[-1][0]}.shape[0] if hasattr(self.{class_params[-1][0]}, 'shape') else 2")
            lines.append(f"        for {loop_var} in range(n):")
            for stmt in loop_body:
                if stmt and stmt[0] == "loop_assign":
                    var_name = stmt[1]
                    expr = stmt[2]
                    expr_code = ast_to_torch_expr(expr)
                    expr_code = replace_class_params(expr_code, class_params)
                    lines.append(f"            {var_name} = {expr_code}")

        # Generate return
        body_code = ast_to_torch_expr(body)
        body_code = replace_class_params(body_code, class_params)
        lines.append(f"        return {body_code}")

        # loss method if present
        if has_loss and loss_body:
            loss_params = class_def.get("loss_params", [("y", "ℝ"), ("target", "ℝ")])
            loss_param_names = [p[0] for p in loss_params]

            # Check if loss uses grad - if so, we need to also pass input x
            loss_uses_grad = ast_uses_func(loss_body, "grad")

            if loss_uses_grad and lambda_param_names:
                # Add the input parameter (x) to loss params
                input_param = lambda_param_names[0]  # typically 'x'
                lines.append(f"")
                lines.append(f"    def loss(self, {', '.join(loss_param_names)}, {input_param}):")
            else:
                lines.append(f"")
                lines.append(f"    def loss(self, {', '.join(loss_param_names)}):")

            loss_code = ast_to_torch_expr(loss_body)
            lines.append(f"        return {loss_code}")

        return "\n".join(lines)

    def generate_statement(stmt):
        """Generate PyTorch code for a program statement."""
        if stmt is None:
            return None

        op = stmt[0]

        if op == "decl":
            name = stmt[1]
            type_spec = stmt[2]
            expr = stmt[3]
            expr_code = ast_to_torch_expr(expr)
            # Variables used as grad targets need to be tensors with requires_grad
            if name in grad_target_vars and type_spec == "ℝ":
                return f"{name} = torch.tensor({expr_code}, requires_grad=True)"
            return f"{name} = {expr_code}"

        elif op == "assign":
            name = stmt[1]
            expr = stmt[2]
            expr_code = ast_to_torch_expr(expr)
            return f"{name} = {expr_code}"

        elif op == "expr":
            expr = stmt[1]
            expr_code = ast_to_torch_expr(expr)
            # Don't wrap side-effect-only calls in physika_print
            if isinstance(expr, tuple) and expr[0] == "call" and expr[1] in ("simulate", "animate"):
                return expr_code
            return f"physika_print({expr_code})"

        elif op == "func_def":
            return None  # Already generated

        elif op == "class_def":
            return None  # Already generated

        elif op == "for_loop":
            # For loop: ("for_loop", loop_var, body_statements, indexed_arrays[, lineno])
            loop_var = stmt[1]
            body_statements = stmt[2]
            indexed_arrays = stmt[3]
            lines = []
            # Use first indexed array to get length
            if indexed_arrays:
                arr_name = indexed_arrays[0]
                lines.append(f"for {loop_var} in range(len({arr_name})):")
            else:
                lines.append(f"for {loop_var} in range(n):  # TODO: determine n")

            # Set current loop var for imaginary -> loop var conversion
            old_loop_var = current_loop_var[0]
            current_loop_var[0] = loop_var

            for body_stmt in body_statements:
                if body_stmt is None:
                    continue
                body_op = body_stmt[0]
                if body_op == "for_assign":
                    _, var_name, expr = body_stmt
                    expr_code = ast_to_torch_expr(expr)
                    lines.append(f"    {var_name} = {expr_code}")
                elif body_op == "for_pluseq":
                    _, var_name, expr = body_stmt
                    expr_code = ast_to_torch_expr(expr)
                    lines.append(f"    {var_name} = {var_name} + {expr_code}")
                elif body_op == "for_call":
                    _, func_name, arg_asts = body_stmt
                    arg_strs = [ast_to_torch_expr(arg) for arg in arg_asts]
                    lines.append(f"    {func_name}({', '.join(arg_strs)})")

            # Restore old loop var
            current_loop_var[0] = old_loop_var
            return "\n".join(lines)

        return f"# Unknown: {stmt}"

    # Check if solve is used anywhere in the AST
    def ast_uses_solve(node):
        if not isinstance(node, (tuple, list)):
            return False
        if isinstance(node, tuple) and len(node) >= 2:
            if node[0] == "call" and node[1] == "solve":
                return True
            return any(ast_uses_solve(child) for child in node[1:] if isinstance(child, (tuple, list)))
        if isinstance(node, list):
            return any(ast_uses_solve(item) for item in node)
        return False

    needs_solve = any(ast_uses_solve(stmt) for stmt in unified_ast["program"])
    for func_def in unified_ast["functions"].values():
        if ast_uses_solve(func_def.get("body")) or any(ast_uses_solve(s) for s in func_def.get("statements", [])):
            needs_solve = True
            break

    # Check if train/evaluate/grad are used
    def ast_uses_func(node, func_name):
        if not isinstance(node, (tuple, list)):
            return False
        if isinstance(node, tuple) and len(node) >= 2:
            if node[0] == "call" and node[1] == func_name:
                return True
            if node[0] == "call_index" and node[1] == func_name:
                return True
            return any(ast_uses_func(child, func_name) for child in node[1:] if isinstance(child, (tuple, list)))
        if isinstance(node, list):
            return any(ast_uses_func(item, func_name) for item in node)
        return False

    needs_train = any(ast_uses_func(stmt, "train") for stmt in unified_ast["program"])
    needs_evaluate = any(ast_uses_func(stmt, "evaluate") for stmt in unified_ast["program"])
    needs_simulate = any(ast_uses_func(stmt, "simulate") for stmt in unified_ast["program"])
    needs_animate = any(ast_uses_func(stmt, "animate") for stmt in unified_ast["program"])

    # Collect variables used as differentiation targets in grad() calls
    # These need to be tensors with requires_grad=True from declaration
    grad_target_vars = set()
    def collect_grad_targets(node):
        if not isinstance(node, (tuple, list)):
            return
        if isinstance(node, tuple) and len(node) >= 2:
            if node[0] == "call" and node[1] == "grad" and len(node) >= 3:
                args = node[2]
                if len(args) >= 2 and isinstance(args[1], tuple) and args[1][0] == "var":
                    grad_target_vars.add(args[1][1])
            for child in node[1:]:
                if isinstance(child, (tuple, list)):
                    collect_grad_targets(child)
        elif isinstance(node, list):
            for item in node:
                collect_grad_targets(item)
    for stmt in unified_ast["program"]:
        collect_grad_targets(stmt)

    # Check for grad usage in classes (for loss functions) and program statements
    needs_grad = False
    for class_def in unified_ast["classes"].values():
        if ast_uses_func(class_def.get("loss_body"), "grad"):
            needs_grad = True
            break
        if ast_uses_func(class_def.get("body"), "grad"):
            needs_grad = True
            break
    # Also check program statements for grad usage
    if not needs_grad:
        for stmt in unified_ast["program"]:
            if ast_uses_func(stmt, "grad"):
                needs_grad = True
                break

    # Generate code header
    code_lines.append("import torch")
    code_lines.append("import torch.nn as nn")
    code_lines.append("import torch.optim as optim")
    if needs_solve:
        code_lines.append("import re")
    code_lines.append("")

    # Physika print helper (value ∈ type format)
    code_lines.append("# === Physika Print Helper ===")
    code_lines.append('''def physika_print(value):
    def _from_torch(v):
        if not isinstance(v, torch.Tensor):
            if isinstance(v, complex):
                if abs(v.imag) < 1e-10:
                    return v.real
                return v
            return v
        if v.numel() == 1:
            val = v.item()
            if isinstance(val, complex) and abs(val.imag) < 1e-10:
                return val.real
            return val
        return v.detach().tolist()

    def _infer_type(v):
        if isinstance(v, complex):
            if v.imag == 0:
                return "\u211d"
            return "\u2102"
        if isinstance(v, torch.Tensor) and v.is_complex():
            if v.imag.abs().max() < 1e-10:
                return "\u211d"
            return "\u2102"
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return "\u211d"
            if v.dim() == 1:
                return f"\u211d[{v.shape[0]}]"
            dims = ",".join(str(d) for d in v.shape)
            return f"\u211d[{dims}]"
        if isinstance(v, (int, float)):
            return "\u211d"
        if isinstance(v, list):
            shape = []
            current = v
            while isinstance(current, list) and len(current) > 0:
                shape.append(len(current))
                current = current[0]
            return f"\u211d[{','.join(str(d) for d in shape)}]"
        if isinstance(v, nn.Module):
            return type(v).__name__
        return str(type(v).__name__)

    display = _from_torch(value)
    type_str = _infer_type(value)
    print(f"{display} \u2208 {type_str}")
''')
    code_lines.append("")

    # Generate solve helper if needed
    if needs_solve:
        code_lines.append("# === Equation Solver ===")
        code_lines.append('''def solve(*equations, **known_vars):
    """
    Solve a system of linear equations.
    equations: strings like 'x0 = a + b', 'v0 = i * omega * a - i * omega * b'
    known_vars: dict of known variable values
    Returns: tuple of solved values in alphabetical order of unknowns
    """
    parsed = []
    for eq in equations:
        lhs, rhs = eq.split('=')
        parsed.append((lhs.strip(), rhs.strip()))

    # Find unknowns (variables in RHS not in known_vars)
    all_rhs_vars = set()
    for lhs, rhs in parsed:
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs)
        all_rhs_vars.update(tokens)

    special = {'i', 'exp', 'sin', 'cos', 'sqrt'}
    unknowns = sorted([v for v in all_rhs_vars if v not in special and v not in known_vars])

    n = len(unknowns)
    use_complex = any('i' in rhs for _, rhs in parsed)

    dtype = torch.complex64 if use_complex else torch.float32
    A = torch.zeros((n, n), dtype=dtype)
    b = torch.zeros(n, dtype=dtype)

    # Build coefficient matrix
    for i, (lhs, rhs) in enumerate(parsed):
        # Get LHS value
        b[i] = known_vars[lhs]

        # Extract coefficients for each unknown
        for j, u in enumerate(unknowns):
            # Simple coefficient extraction (handles i * omega * var patterns)
            coeff = 0
            # Check for patterns like: "coeff * var" or "var"
            pattern = rf'([+-]?\\s*(?:[\\d.]*\\s*\\*\\s*)?(?:i\\s*\\*\\s*)?(?:[a-zA-Z_][a-zA-Z0-9_]*\\s*\\*\\s*)*)\\b{u}\\b'
            matches = re.finditer(pattern, rhs)
            for m in matches:
                coeff_str = m.group(1).strip()
                if not coeff_str or coeff_str == '+':
                    coeff += 1
                elif coeff_str == '-':
                    coeff += -1
                else:
                    # Evaluate the coefficient
                    coeff_str = coeff_str.rstrip('* ')
                    coeff_str = coeff_str.replace('i', '1j')
                    for var, val in known_vars.items():
                        coeff_str = re.sub(rf'\\b{var}\\b', str(complex(val) if use_complex else float(val)), coeff_str)
                    try:
                        coeff += eval(coeff_str)
                    except:
                        coeff += 1
            A[i, j] = coeff

    # Solve using torch.linalg.solve
    solution = torch.linalg.solve(A, b)
    return tuple(solution[i] for i in range(n))
''')
        code_lines.append("")

    # Generate train helper if needed
    if needs_train:
        code_lines.append("# === Training Helper ===")
        if needs_grad:
            # Train helper that passes input x to loss (for physics-informed loss)
            code_lines.append('''def train(model, X, y, epochs, lr):
    """
    Train a neural network model with physics-informed loss.
    model: nn.Module instance
    X: input data tensor [n_samples, n_features]
    y: target tensor [n_samples]
    epochs: number of training epochs
    lr: learning rate
    Returns: trained model
    """
    import copy
    import inspect
    trained_model = copy.deepcopy(model)

    # Make parameters require gradients
    for param in trained_model.parameters():
        param.requires_grad_(True)

    optimizer = optim.SGD(trained_model.parameters(), lr=lr)

    # Check if loss method takes input x (3 params: pred, target, x)
    loss_takes_input = False
    if hasattr(trained_model, 'loss'):
        sig = inspect.signature(trained_model.loss)
        loss_takes_input = len(sig.parameters) == 3

    epochs = int(epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)

        for i in range(X.shape[0]):
            x_i = X[i].clone().requires_grad_(True)
            y_i = y[i]
            pred = trained_model(x_i)

            # Use model's loss method if available
            if hasattr(trained_model, 'loss'):
                if loss_takes_input:
                    loss_i = trained_model.loss(pred, y_i, x_i)
                else:
                    loss_i = trained_model.loss(pred, y_i)
            else:
                loss_i = (pred - y_i) ** 2

            total_loss = total_loss + loss_i

        total_loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")

    return trained_model
''')
        else:
            # Standard train helper
            code_lines.append('''def train(model, X, y, epochs, lr):
    """
    Train a neural network model.
    model: nn.Module instance
    X: input data tensor [n_samples, n_features]
    y: target tensor [n_samples]
    epochs: number of training epochs
    lr: learning rate
    Returns: trained model
    """
    import copy
    trained_model = copy.deepcopy(model)

    # Make parameters require gradients
    for param in trained_model.parameters():
        param.requires_grad_(True)

    optimizer = optim.SGD(trained_model.parameters(), lr=lr)

    epochs = int(epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)

        for i in range(X.shape[0]):
            x_i = X[i]
            y_i = y[i]
            pred = trained_model(x_i)

            # Use model's loss method if available
            if hasattr(trained_model, 'loss'):
                loss_i = trained_model.loss(pred, y_i)
            else:
                loss_i = (pred - y_i) ** 2

            total_loss = total_loss + loss_i

        total_loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")

    return trained_model
''')
        code_lines.append("")

    # Generate evaluate helper if needed
    if needs_evaluate:
        code_lines.append("# === Evaluation Helper ===")
        if needs_grad:
            # Evaluate helper that passes input x to loss (for physics-informed loss)
            code_lines.append('''def evaluate(model, X, y):
    """
    Evaluate a model on data with physics-informed loss.
    model: nn.Module instance
    X: input data tensor [n_samples, n_features]
    y: target tensor [n_samples]
    Returns: mean loss
    """
    import inspect
    total_loss = 0.0
    n_samples = X.shape[0]

    # Check if loss method takes input x (3 params: pred, target, x)
    loss_takes_input = False
    if hasattr(model, 'loss'):
        sig = inspect.signature(model.loss)
        loss_takes_input = len(sig.parameters) == 3

    for i in range(n_samples):
        x_i = X[i].clone().requires_grad_(True)
        y_i = y[i]
        pred = model(x_i)

        # Use model's loss method if available
        if hasattr(model, 'loss'):
            if loss_takes_input:
                loss_i = model.loss(pred, y_i, x_i)
            else:
                loss_i = model.loss(pred, y_i)
        else:
            loss_i = (pred - y_i) ** 2

        if isinstance(loss_i, torch.Tensor):
            loss_i = loss_i.item()
        total_loss += loss_i

    return total_loss / n_samples
''')
        else:
            code_lines.append('''def evaluate(model, X, y):
    """
    Evaluate a model on data.
    model: nn.Module instance
    X: input data tensor [n_samples, n_features]
    y: target tensor [n_samples]
    Returns: mean loss
    """
    total_loss = 0.0
    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]
            pred = model(x_i)

            # Use model's loss method if available
            if hasattr(model, 'loss'):
                loss_i = model.loss(pred, y_i)
            else:
                loss_i = (pred - y_i) ** 2

            if isinstance(loss_i, torch.Tensor):
                loss_i = loss_i.item()
            total_loss += loss_i

    return total_loss / n_samples
''')
        code_lines.append("")

    # Generate compute_grad helper if needed
    if needs_grad:
        code_lines.append("# === Gradient Helper ===")
        code_lines.append('''def compute_grad(output, input):
    """
    Compute gradient of output with respect to input using autograd.
    output: scalar tensor (the value to differentiate)
    input: tensor with requires_grad=True (the variable to differentiate with respect to)
    Returns: gradient tensor
    """
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(float(input), requires_grad=True)
    if not input.requires_grad:
        input = input.clone().requires_grad_(True)
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output, dtype=torch.float32)
    grads = torch.autograd.grad(output, input, create_graph=True, retain_graph=True)
    return grads[0]
''')
        code_lines.append("")

    # Generate simulate helper if needed
    if needs_simulate:
        code_lines.append("# === Simulate Helper ===")
        code_lines.append('''def simulate(model, x0, nsteps, dt):
    """
    Run an ODE solver model for nsteps iterations, collect trajectory, and plot.
    model: nn.Module instance (single-step solver)
    x0: initial state tensor
    nsteps: number of time steps
    dt: time step size
    """
    import matplotlib.pyplot as plt
    x = torch.as_tensor(x0).float()
    nsteps = int(nsteps)
    dt_val = float(dt) if isinstance(dt, torch.Tensor) else float(dt)
    trajectory = [x.detach().clone()]
    with torch.no_grad():
        for i in range(nsteps):
            x = model(x)
            trajectory.append(x.detach().clone())
    states = torch.stack(trajectory)
    t = torch.arange(states.shape[0]).float() * dt_val
    if states.dim() == 1 or states.shape[-1] == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(t.numpy(), states.squeeze().numpy())
        plt.ylabel("x")
        plt.xlabel("Time (s)")
        plt.title("Physika")
        plt.grid(True)
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        labels = [f"x[{j}]" for j in range(states.shape[1])]
        for j in range(states.shape[1]):
            ax1.plot(t.numpy(), states[:, j].numpy(), label=labels[j])
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("State")
        ax1.set_title("Time Evolution")
        ax1.grid(True)
        ax2.plot(states[:, 1].numpy(), states[:, 0].numpy(), linewidth=0.8)
        ax2.plot(states[0, 1].item(), states[0, 0].item(), 'ro', markersize=6, label='Start')
        ax2.set_xlabel("x[1]")
        ax2.set_ylabel("x[0]")
        ax2.set_title("Phase Space")
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal', adjustable='datalim')
        fig.suptitle("Physika", fontsize=14)
        plt.tight_layout()
        plt.show()

    # --- PyVista 3D pendulum animation ---
    state_dim = states.shape[-1] if states.dim() > 1 else 1
    if state_dim in (2, 4):
        try:
            import pyvista as pv
            import numpy as np
            import time as time_module

            total_frames = states.shape[0]
            step_size = max(1, total_frames // 2000)
            idx = list(range(0, total_frames, step_size))
            sub_states = states[idx].numpy()
            sub_t = t[idx].numpy()

            if state_dim == 2:
                L = 1.0
                theta_vals = sub_states[:, 0]
                xs = L * np.sin(theta_vals)
                ys = -L * np.cos(theta_vals)
                title_str = "Physika \\n\\nSimple Pendulum Animation"
                rod_color = "black"
                bob_radius = 0.06
            else:
                r_vals = sub_states[:, 0]
                theta_vals = sub_states[:, 1]
                xs = r_vals * np.sin(theta_vals)
                ys = -r_vals * np.cos(theta_vals)
                title_str = "Physika \\n\\nSpring Pendulum Animation"
                rod_color = "black"
                bob_radius = 0.07

            pad = 0.3
            x_range = float(np.max(np.abs(xs))) + pad
            y_min = float(np.min(ys)) - pad
            y_max = max(float(np.max(ys)) + pad, pad)
            scene_range = max(x_range, abs(y_min), abs(y_max))

            plotter = pv.Plotter()
            plotter.add_title(title_str, font_size=20, font="times", shadow=True)

            # XYZ reference axes (gray dashed)
            axis_len = scene_range * 1.2
            for axis_pt in [((axis_len, 0, 0), "X"), ((0, axis_len, 0), "Y"), ((0, 0, axis_len), "Z")]:
                end, label = axis_pt
                neg = tuple(-c for c in end)
                line = pv.Line(neg, end, resolution=60)
                plotter.add_mesh(line, color="gray", style="wireframe", line_width=1, opacity=0.5)
                plotter.add_point_labels([end], [label], font_size=14, text_color="gray", shadow=False, shape=None)

            # Pivot
            pivot = pv.Sphere(radius=0.025, center=(0, 0, 0))
            plotter.add_mesh(pivot, color="red")

            # Bob
            bob = pv.Sphere(radius=bob_radius, center=(xs[0], ys[0], 0))
            plotter.add_mesh(bob, color="blue")

            # Rod
            rod = pv.Line((0, 0, 0), (xs[0], ys[0], 0))
            rod_actor = plotter.add_mesh(rod, color=rod_color, line_width=3)

            # Trail
            trail_len = 80
            trail_pts = np.zeros((trail_len, 3))
            trail_pts[:, 0] = xs[0]
            trail_pts[:, 1] = ys[0]
            trail_line = pv.Spline(trail_pts, n_points=trail_len)
            trail_actor = plotter.add_mesh(trail_line, color="brown", line_width=2, opacity=0.6)

            # Camera
            plotter.camera_position = [(0, 0, 3 * scene_range),
                                       (0, (y_min + y_max) / 2, 0),
                                       (0, 1, 0)]

            anim_state = {"paused": False, "running": True}

            def on_space():
                anim_state["paused"] = not anim_state["paused"]
            def on_quit():
                anim_state["running"] = False

            plotter.add_key_event("space", on_space)
            plotter.add_key_event("q", on_quit)

            if state_dim == 2:
                info = (f"t = {sub_t[0]:.3f}\\n"
                        f"\\u03b8 = {sub_states[0, 0]:.4f}\\n"
                        f"\\u03c9 = {sub_states[0, 1]:.4f}\\n"
                        f"[SPACE: pause | Q: quit]")
            else:
                info = (f"t = {sub_t[0]:.3f}\\n"
                        f"r = {sub_states[0, 0]:.4f}  \\u03b8 = {sub_states[0, 1]:.4f}\\n"
                        f"dr = {sub_states[0, 2]:.4f}  d\\u03b8 = {sub_states[0, 3]:.4f}\\n"
                        f"[SPACE: pause | Q: quit]")

            text_actor = plotter.add_text(info, position=(10, 10), font_size=13, font="times")

            plotter.show(auto_close=False, interactive_update=True)

            trail_history = []
            while anim_state["running"]:
                trail_history.clear()
                for i in range(len(xs)):
                    if not anim_state["running"]:
                        break
                    while anim_state["paused"] and anim_state["running"]:
                        plotter.update()
                        time_module.sleep(0.05)
                    if not anim_state["running"]:
                        break

                    bx, by = float(xs[i]), float(ys[i])
                    bob.points = pv.Sphere(radius=bob_radius, center=(bx, by, 0)).points

                    new_rod = pv.Line((0, 0, 0), (bx, by, 0))
                    plotter.remove_actor(rod_actor)
                    rod_actor = plotter.add_mesh(new_rod, color=rod_color, line_width=3)

                    trail_history.append([bx, by, 0.0])
                    if len(trail_history) > trail_len:
                        trail_history.pop(0)
                    if len(trail_history) >= 2:
                        tp = np.array(trail_history)
                        new_trail = pv.Spline(tp, n_points=len(tp))
                        plotter.remove_actor(trail_actor)
                        trail_actor = plotter.add_mesh(new_trail, color="cyan", line_width=2, opacity=0.6)

                    pause_status = "[PAUSED]" if anim_state["paused"] else "[SPACE: pause | Q: quit]"
                    if state_dim == 2:
                        info = (f"t = {sub_t[i]:.3f}\\n"
                                f"\\u03b8 = {sub_states[i, 0]:.4f}\\n"
                                f"\\u03c9 = {sub_states[i, 1]:.4f}\\n"
                                f"{pause_status}")
                    else:
                        info = (f"t = {sub_t[i]:.3f}\\n"
                                f"r = {sub_states[i, 0]:.4f}  \\u03b8 = {sub_states[i, 1]:.4f}\\n"
                                f"dr = {sub_states[i, 2]:.4f}  d\\u03b8 = {sub_states[i, 3]:.4f}\\n"
                                f"{pause_status}")
                    text_actor.SetInput(info)

                    plotter.update()
                    time_module.sleep(0.02)

            plotter.close()
        except ImportError:
            pass
        except Exception as e:
            print(f"[simulate] PyVista animation error: {e}")
''')
        code_lines.append("")

    # Generate animate helper if needed
    if needs_animate:
        code_lines.append("# === Animate Helper ===")
        code_lines.append('''def animate(func, *args):
    """
    Animate a Physika function over time.
    animate(func, fixed_args..., time_min, time_max, [n_points])
    n_points is optional, defaults to 200.
    """
    import numpy as np
    import time as time_module

    if len(args) < 2:
        print("[animate] Need at least time_min and time_max")
        return

    # Detect if n_points was provided (last arg is integer-like and >= 10)
    last_arg = args[-1]
    if isinstance(last_arg, torch.Tensor):
        last_arg_val = last_arg.item()
    else:
        last_arg_val = last_arg

    def is_integer_like(val):
        if isinstance(val, int):
            return True
        if isinstance(val, float):
            return val == int(val)
        return False

    is_n_points_provided = is_integer_like(last_arg_val) and last_arg_val >= 10

    if is_n_points_provided:
        time_min = args[-3]
        time_max = args[-2]
        n_points = int(last_arg_val)
        fixed_args = list(args[:-3])
    else:
        time_min = args[-2]
        time_max = args[-1]
        n_points = 200
        fixed_args = list(args[:-2])

    if isinstance(time_min, torch.Tensor):
        time_min = time_min.item()
    if isinstance(time_max, torch.Tensor):
        time_max = time_max.item()

    time_vals = np.linspace(float(time_min), float(time_max), n_points)
    x_values = []

    for t in time_vals:
        call_args = []
        for a in fixed_args[:2]:
            if isinstance(a, torch.Tensor):
                call_args.append(a)
            else:
                call_args.append(torch.tensor(float(a), requires_grad=True))
        call_args.append(torch.tensor(float(t), requires_grad=True))
        for a in fixed_args[2:]:
            if isinstance(a, torch.Tensor):
                call_args.append(a)
            else:
                call_args.append(torch.tensor(float(a), requires_grad=True))

        result = func(*call_args)

        if isinstance(result, torch.Tensor):
            if result.is_complex():
                result = result.real
            x_values.append(result.item())
        elif isinstance(result, complex):
            x_values.append(result.real)
        else:
            x_values.append(float(result))

    x_values = np.array(x_values)

    # Compute velocity via numerical differentiation: v = dx/dt
    dt = (float(time_max) - float(time_min)) / (n_points - 1)
    v_values = np.gradient(x_values, dt)

    # Try PyVista first, then matplotlib
    try:
        import pyvista as pv
        HAS_PYVISTA = True
    except ImportError:
        HAS_PYVISTA = False

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    if HAS_PYVISTA:
        plotter = pv.Plotter()
        plotter.add_title(
            "Physika \\n\\nHarmonic Oscillator Animation",
            font_size=24,
            font="times",
            shadow=True
        )

        sphere = pv.Sphere(radius=0.1, center=(x_values[0], 0, 0))
        plotter.add_mesh(sphere, color='blue')
        plotter.add_mesh(pv.Line((-2, 0, 0), (2, 0, 0)), color='black', line_width=3)

        # Red dot indicating initial position x0
        x0_marker = pv.Sphere(radius=0.03, center=(x_values[0], 0, 0))
        plotter.add_mesh(x0_marker, color='red')

        plotter.camera_position = [(0, 5, 0), (0, 0, 0), (0, 0, 1)]

        # State for pause and loop
        anim_state = {"paused": False, "running": True}

        def on_key_press(key):
            if key == "space":
                anim_state["paused"] = not anim_state["paused"]
            elif key == "q" or key == "Escape":
                anim_state["running"] = False

        plotter.add_key_event("space", lambda: on_key_press("space"))
        plotter.add_key_event("q", lambda: on_key_press("q"))

        text_actor = plotter.add_text(
            f"t = {time_vals[0]:.3f}\\nx = {x_values[0]:.4f}\\nv = {v_values[0]:.4f}\\n[SPACE: pause | Q: quit]",
            position=(10, 10), font_size=15, font="times"
        )

        plotter.show(auto_close=False, interactive_update=True)

        while anim_state["running"]:
            for i, x in enumerate(x_values):
                if not anim_state["running"]:
                    break
                while anim_state["paused"] and anim_state["running"]:
                    plotter.update()
                    time_module.sleep(0.05)
                if not anim_state["running"]:
                    break
                sphere.points = pv.Sphere(radius=0.1, center=(x, 0, 0)).points
                pause_status = "[PAUSED]" if anim_state["paused"] else "[SPACE: pause | Q: quit]"
                text_actor.SetInput(
                    f"t = {time_vals[i]:.3f}\\nx = {x_values[i]:.4f}\\nv = {v_values[i]:.4f}\\n{pause_status}"
                )
                plotter.update()
                time_module.sleep(0.03)

        plotter.close()

    elif HAS_MATPLOTLIB:
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='black', linewidth=2)
        ax.set_title("Harmonic Oscillator Animation [SPACE: pause/resume | R: reset]")

        mass, = ax.plot([], [], 'bo', markersize=20)
        spring, = ax.plot([], [], 'gray', linewidth=2)

        ax.plot([x_values[0]], [0], 'ro', markersize=8, label=f'x0 = {x_values[0]:.2f}')

        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontfamily='monospace',
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        anim_state = {"paused": False, "frame": 0}
        ani_ref = [None]

        def init():
            mass.set_data([], [])
            spring.set_data([], [])
            info_text.set_text('')
            return mass, spring, info_text

        def anim(i):
            if anim_state["paused"]:
                i = anim_state["frame"]
            else:
                anim_state["frame"] = i
            mass.set_data([x_values[i]], [0])
            spring.set_data([0, x_values[i]], [0, 0])
            pause_str = " [PAUSED]" if anim_state["paused"] else ""
            info_text.set_text(f't = {time_vals[i]:.3f}{pause_str}\\nx = {x_values[i]:.4f}\\nv = {v_values[i]:.4f}')
            return mass, spring, info_text

        def on_key(event):
            if event.key == ' ':
                anim_state["paused"] = not anim_state["paused"]
                if not anim_state["paused"] and ani_ref[0] is not None:
                    ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()
            elif event.key == 'r':
                anim_state["frame"] = 0
                anim_state["paused"] = False
                if ani_ref[0] is not None:
                    ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()

        fig.canvas.mpl_connect('key_press_event', on_key)

        ani = FuncAnimation(fig, anim, init_func=init, frames=len(x_values),
                            interval=30, blit=True, repeat=True)
        ani_ref[0] = ani
        plt.show()
    else:
        print("[animate] No visualization backend available (install pyvista or matplotlib)")
''')
        code_lines.append("")

    # Generate functions
    if unified_ast["functions"]:
        code_lines.append("# === Functions ===")
        for name, func_def in unified_ast["functions"].items():
            code_lines.append(generate_function(name, func_def))
            code_lines.append("")

    # Generate classes
    if unified_ast["classes"]:
        code_lines.append("# === Classes ===")
        for name, class_def in unified_ast["classes"].items():
            code_lines.append(generate_class(name, class_def))
            code_lines.append("")

    # Generate program statements
    code_lines.append("# === Program ===")
    for stmt in unified_ast["program"]:
        stmt_code = generate_statement(stmt)
        if stmt_code:
            code_lines.append(stmt_code)

    # Join all code
    generated_code = "\n".join(code_lines)

    if print_code:
        print("\n=== Physika generated Pytorch code ===")
        print(generated_code)
        print("=== End Pytorch code ===\n")

    return generated_code
