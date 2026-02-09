# ======================================================
# UNIFIED AST BUILDER
# ======================================================

def build_unified_ast(program_ast, symbol_table):
    """
    Build a unified AST structure combining definitions and program statements.

    Returns:
        {
            "functions": {name: func_def, ...},
            "classes": {name: class_def, ...},
            "program": [stmt, ...]
        }
    """
    unified = {
        "functions": {},
        "classes": {},
        "program": []
    }

    # Extract functions and classes from symbol table
    for name, entry in symbol_table.items():
        if entry["type"] == "function":
            unified["functions"][name] = entry["value"]
        elif entry["type"] == "class":
            unified["classes"][name] = entry["value"]

    # Add program statements (filtering out func_def/class_def markers)
    for stmt in program_ast:
        if stmt is not None:
            unified["program"].append(stmt)

    return unified

# ======================================================
# TYPE CHECKER
# ======================================================

class TypeError(Exception):
    """Custom exception for type errors in Physika programs."""
    pass


def type_check(unified_ast):
    """
    Type check the unified AST.

    Args:
        unified_ast: The unified AST from build_unified_ast()

    Returns:
        List of type errors (empty if no errors)

    Raises:
        TypeError: If strict mode and errors found
    """
    errors = []
    type_env = {}  # Variable name -> type
    func_env = {}  # Function name -> (param_types, return_type)
    class_env = {}  # Class name -> class_def
    current_line = [None]  # Use list to allow nonlocal modification

    def add_error(msg):
        """Add an error with current line context if available."""
        if current_line[0] is not None:
            errors.append(f"Line {current_line[0]}: {msg}")
        else:
            errors.append(msg)

    def type_to_str(t):
        """Convert type to readable string."""
        if t is None:
            return "unknown"
        if isinstance(t, str):
            return t
        if isinstance(t, tuple):
            if t[0] == "tensor":
                dims = t[1]
                if len(dims) == 1:
                    return f"ℝ[{dims[0][0]}]"
                else:
                    dim_strs = [str(d[0]) for d in dims]
                    return f"ℝ[{','.join(dim_strs)}]"
            elif t[0] == "func_type":
                return f"({t[1]}) → {type_to_str(t[2])}"
        return str(t)

    def get_shape(t):
        """Extract shape from a tensor type, or None for scalars."""
        if t == "ℝ" or t == "ℕ":
            return None  # Scalar
        if isinstance(t, tuple) and t[0] == "tensor":
            return [d[0] for d in t[1]]
        return None

    def make_tensor_type(shape):
        """Create a tensor type from a shape list."""
        if shape is None or len(shape) == 0:
            return "ℝ"
        return ("tensor", [(d, "invariant") for d in shape])

    def types_compatible(t1, t2):
        """Check if two types are compatible (for assignment or operations)."""
        if t1 is None or t2 is None:
            return True  # Unknown types are compatible with anything
        if t1 == t2:
            return True
        # ℝ and ℕ are compatible
        if t1 in ("ℝ", "ℕ") and t2 in ("ℝ", "ℕ"):
            return True
        # Check tensor shapes
        shape1 = get_shape(t1)
        shape2 = get_shape(t2)
        if shape1 == shape2:
            return True
        return False

    def shapes_broadcast_compatible(s1, s2, allow_scalar_broadcast=False):
        """
        Check if two shapes are compatible for element-wise operations.

        Args:
            s1, s2: Shapes to compare (None means scalar)
            allow_scalar_broadcast: If True, allows scalar * tensor operations

        Returns:
            (result_shape, is_compatible)
        """
        if s1 is None and s2 is None:
            return None, True  # Both scalars
        if s1 == s2:
            return s1, True  # Same shape

        # Strict mode: no broadcasting between scalar and tensor
        if not allow_scalar_broadcast:
            if s1 is None or s2 is None:
                return None, False  # Shape mismatch: scalar vs tensor

        # Allow scalar broadcast only if explicitly enabled
        if allow_scalar_broadcast:
            if s1 is None:
                return s2, True
            if s2 is None:
                return s1, True

        return None, False

    def infer_type(expr, local_env=None):
        """
        Infer the type of an expression.

        Args:
            expr: AST expression node
            local_env: Local type environment (for function bodies)

        Returns:
            Inferred type, or None if cannot be determined
        """
        if local_env is None:
            local_env = {}

        if not isinstance(expr, tuple):
            if isinstance(expr, (int, float)):
                return "ℝ"
            return None

        op = expr[0]

        if op == "num":
            return "ℝ"

        elif op == "var":
            var_name = expr[1]
            if var_name in local_env:
                return local_env[var_name]
            if var_name in type_env:
                return type_env[var_name]
            return None

        elif op == "array":
            elements = expr[1]
            if not elements:
                return ("tensor", [(0, "invariant")])

            def infer_array_shape(arr_node):
                """Recursively infer the shape of a nested array."""
                if not isinstance(arr_node, tuple) or arr_node[0] != "array":
                    return []  # Scalar element

                elems = arr_node[1]
                if not elems:
                    return [0]

                outer_dim = len(elems)

                # Check if elements are nested arrays
                first_elem = elems[0]
                if isinstance(first_elem, tuple) and first_elem[0] == "array":
                    # Get inner shape from first element
                    inner_shape = infer_array_shape(first_elem)

                    # Verify all elements have the same shape
                    for i, elem in enumerate(elems):
                        if not isinstance(elem, tuple) or elem[0] != "array":
                            add_error(f"Inconsistent array nesting at index {i}: expected array, got scalar")
                            return None
                        elem_shape = infer_array_shape(elem)
                        if elem_shape != inner_shape:
                            add_error(f"Inconsistent array shape at index {i}: {elem_shape} vs {inner_shape}")
                            return None

                    return [outer_dim] + inner_shape
                else:
                    # Leaf level - all elements should be scalars
                    for i, elem in enumerate(elems):
                        if isinstance(elem, tuple) and elem[0] == "array":
                            add_error(f"Inconsistent nesting at index {i}: expected scalar, got array")
                            return None
                    return [outer_dim]

            shape = infer_array_shape(expr)
            if shape is None:
                return None
            return ("tensor", [(d, "invariant") for d in shape])

        elif op == "index":
            var_name = expr[1]
            var_type = local_env.get(var_name) or type_env.get(var_name)
            if var_type is None:
                return None
            shape = get_shape(var_type)
            if shape is None:
                add_error(f"Cannot index scalar '{var_name}'")
                return None
            if len(shape) == 1:
                return "ℝ"  # Indexing 1D array gives scalar
            else:
                # Indexing multi-dim array gives sub-array
                return make_tensor_type(shape[1:])

        elif op == "slice":
            var_name = expr[1]
            start_expr = expr[2]
            end_expr = expr[3]
            var_type = local_env.get(var_name) or type_env.get(var_name)
            if var_type is None:
                return None
            shape = get_shape(var_type)
            if shape is None:
                add_error(f"Cannot slice scalar '{var_name}'")
                return None

            # Try to compute slice length
            start_val = None
            end_val = None
            if isinstance(start_expr, tuple) and start_expr[0] == "num":
                start_val = int(start_expr[1])
            if isinstance(end_expr, tuple) and end_expr[0] == "num":
                end_val = int(end_expr[1])

            if start_val is not None and end_val is not None:
                # Physika uses inclusive end, so length is end - start + 1
                slice_len = end_val - start_val + 1
                if len(shape) == 1:
                    return ("tensor", [(slice_len, "invariant")])
                else:
                    new_shape = [slice_len] + shape[1:]
                    return make_tensor_type(new_shape)
            return None  # Cannot determine slice length

        elif op in ("add", "sub"):
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            # Strict shape matching for add/sub - no scalar broadcasting
            result_shape, ok = shapes_broadcast_compatible(left_shape, right_shape, allow_scalar_broadcast=False)
            if not ok:
                add_error(f"Shape mismatch in {op}: {type_to_str(left_type)} vs {type_to_str(right_type)}")
                return None
            return make_tensor_type(result_shape)

        elif op == "mul":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            # Allow scalar multiplication with tensors
            result_shape, ok = shapes_broadcast_compatible(left_shape, right_shape, allow_scalar_broadcast=True)
            if not ok:
                add_error(f"Shape mismatch in multiplication: {type_to_str(left_type)} vs {type_to_str(right_type)}")
                return None
            return make_tensor_type(result_shape)

        elif op == "div":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            # Division by scalar is always ok
            right_shape = get_shape(right_type)
            if right_shape is not None:
                left_shape = get_shape(left_type)
                if left_shape != right_shape:
                    add_error(f"Element-wise division requires matching shapes: {type_to_str(left_type)} vs {type_to_str(right_type)}")
            return left_type

        elif op == "matmul":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            if left_shape is None or right_shape is None:
                # Scalar matmul - treat as regular mul
                return left_type if right_shape is None else right_type

            # Matrix multiplication dimension check
            if len(left_shape) == 1 and len(right_shape) == 1:
                # Vector dot product: (n,) @ (n,) -> scalar
                if left_shape[0] != right_shape[0]:
                    add_error(f"Vector dot product dimension mismatch: {left_shape[0]} vs {right_shape[0]}")
                return "ℝ"
            elif len(left_shape) == 2 and len(right_shape) == 1:
                # Matrix-vector: (m,n) @ (n,) -> (m,)
                if left_shape[1] != right_shape[0]:
                    add_error(f"Matrix-vector multiplication dimension mismatch: {left_shape[1]} vs {right_shape[0]}")
                return make_tensor_type([left_shape[0]])
            elif len(left_shape) == 1 and len(right_shape) == 2:
                # Vector-matrix: (m,) @ (m,n) -> (n,)
                if left_shape[0] != right_shape[0]:
                    add_error(f"Vector-matrix multiplication dimension mismatch: {left_shape[0]} vs {right_shape[0]}")
                return make_tensor_type([right_shape[1]])
            elif len(left_shape) == 2 and len(right_shape) == 2:
                # Matrix-matrix: (m,k) @ (k,n) -> (m,n)
                if left_shape[1] != right_shape[0]:
                    add_error(f"Matrix multiplication dimension mismatch: {left_shape[1]} vs {right_shape[0]}")
                return make_tensor_type([left_shape[0], right_shape[1]])

            return None

        elif op == "pow":
            left_type = infer_type(expr[1], local_env)
            # Power typically returns same shape as base
            return left_type

        elif op == "neg":
            return infer_type(expr[1], local_env)

        elif op == "call":
            func_name = expr[1]
            args = expr[2]

            # Built-in functions
            if func_name in ("exp", "log", "sin", "cos", "sqrt", "abs", "tanh"):
                if args:
                    return infer_type(args[0], local_env)
                return "ℝ"
            elif func_name == "sum":
                return "ℝ"  # Sum reduces to scalar
            elif func_name in ("real", "imag"):
                return "ℝ"
            elif func_name == "grad":
                # grad returns gradient with same shape as input
                if len(args) >= 2:
                    return infer_type(args[1], local_env)
                return None
            elif func_name == "solve":
                return None  # Solve returns tuple, type depends on equations
            elif func_name == "train":
                # train returns the same instance type as its first argument
                if args:
                    return infer_type(args[0], local_env)
                return None
            elif func_name == "evaluate":
                return "ℝ"  # evaluate returns a scalar loss

            # User-defined function
            if func_name in func_env:
                _, return_type = func_env[func_name]
                return return_type

            # Class constructor
            if func_name in class_env:
                class_def = class_env[func_name]
                class_params = class_def["class_params"]
                # Check argument count
                if len(args) != len(class_params):
                    add_error(
                        f"Class '{func_name}' expects {len(class_params)} arguments, got {len(args)}"
                    )
                else:
                    # Check each argument type against the declared parameter type
                    for i, ((param_name, param_type), arg_expr) in enumerate(zip(class_params, args)):
                        arg_type = infer_type(arg_expr, local_env)
                        if arg_type is not None and param_type is not None:
                            if not types_compatible(param_type, arg_type):
                                add_error(
                                    f"Type mismatch for parameter '{param_name}' of class '{func_name}': "
                                    f"expected {type_to_str(param_type)}, got {type_to_str(arg_type)}"
                                )
                return ("instance", func_name)

            # Instance call (e.g., net2([1.0, 2.0]) where net2 is an instance)
            var_type = type_env.get(func_name) or (local_env.get(func_name) if local_env else None)
            if isinstance(var_type, tuple) and var_type[0] == "instance":
                class_name = var_type[1]
                if class_name in class_env:
                    class_def = class_env[class_name]
                    lambda_params = class_def["lambda_params"]
                    return_type = class_def.get("return_type")
                    # Check argument count
                    if len(args) != len(lambda_params):
                        add_error(
                            f"Instance of '{class_name}' expects {len(lambda_params)} argument(s), got {len(args)}"
                        )
                    else:
                        # Check each argument type against the lambda parameter type
                        for (param_name, param_type), arg_expr in zip(lambda_params, args):
                            arg_type = infer_type(arg_expr, local_env)
                            if arg_type is not None and param_type is not None:
                                if not types_compatible(param_type, arg_type):
                                    add_error(
                                        f"Type mismatch for parameter '{param_name}' of '{class_name}': "
                                        f"expected {type_to_str(param_type)}, got {type_to_str(arg_type)}"
                                    )
                    return return_type

            return None

        elif op == "call_index":
            func_name = expr[1]
            args = expr[2]
            index = expr[3]

            if func_name == "grad":
                # grad returns gradient vector, indexing gives scalar
                return "ℝ"

            return None

        elif op == "string":
            return "string"

        elif op == "imaginary":
            return "ℂ"  # Complex type

        return None

    def get_line_info(stmt):
        """Extract line number from statement if available."""
        if stmt is None:
            return None
        op = stmt[0]
        # Line number is the last element for statements that have it
        if op == "decl" and len(stmt) >= 5:
            return stmt[4]
        elif op == "assign" and len(stmt) >= 4:
            return stmt[3]
        elif op == "expr" and len(stmt) >= 3:
            return stmt[2]
        elif op == "for_loop" and len(stmt) >= 5:
            return stmt[4]
        return None

    def check_statement(stmt):
        """Check a single statement for type errors."""
        if stmt is None:
            return

        op = stmt[0]
        line = get_line_info(stmt)
        current_line[0] = line  # Set context for expression errors

        if op == "decl":
            # Handle both old format (4 elements) and new format (5 elements with line number)
            name = stmt[1]
            declared_type = stmt[2]
            expr = stmt[3]
            inferred_type = infer_type(expr)

            if inferred_type is not None and declared_type is not None:
                if not types_compatible(declared_type, inferred_type):
                    add_error(
                        f"Type mismatch for '{name}': declared as {type_to_str(declared_type)}, "
                        f"but expression has type {type_to_str(inferred_type)}"
                    )

            # Add to type environment
            type_env[name] = declared_type if declared_type else inferred_type

        elif op == "assign":
            name = stmt[1]
            expr = stmt[2]
            inferred_type = infer_type(expr)

            if name in type_env:
                existing_type = type_env[name]
                if not types_compatible(existing_type, inferred_type):
                    add_error(
                        f"Type mismatch in assignment to '{name}': expected {type_to_str(existing_type)}, "
                        f"got {type_to_str(inferred_type)}"
                    )
            else:
                type_env[name] = inferred_type

        elif op == "expr":
            expr = stmt[1]
            # Just infer the type to check for errors in the expression
            infer_type(expr)

        elif op == "for_loop":
            loop_var = stmt[1]
            body_stmts = stmt[2]
            # Loop variable is an integer index
            type_env[loop_var] = "ℕ"
            for body_stmt in body_stmts:
                check_statement(body_stmt)

    def check_function(name, func_def):
        """Check a function definition for type errors."""
        params = func_def["params"]
        body = func_def["body"]
        statements = func_def.get("statements", [])
        return_type = func_def.get("return_type")

        # Build local environment from parameters
        local_env = {}
        param_types = []
        for param_name, param_type in params:
            local_env[param_name] = param_type
            param_types.append(param_type)

        # Register function signature
        func_env[name] = (param_types, return_type)

        # Check statements in function body
        for stmt in statements:
            if stmt is None:
                continue
            stmt_op = stmt[0]
            if stmt_op == "body_decl":
                _, var_name, var_type, expr = stmt
                inferred = infer_type(expr, local_env)
                if var_type and inferred and not types_compatible(var_type, inferred):
                    errors.append(
                        f"In function '{name}': type mismatch for '{var_name}': "
                        f"declared as {type_to_str(var_type)}, got {type_to_str(inferred)}"
                    )
                local_env[var_name] = var_type if var_type else inferred
            elif stmt_op == "body_assign":
                _, var_name, expr = stmt
                inferred = infer_type(expr, local_env)
                local_env[var_name] = inferred
            elif stmt_op == "body_tuple_unpack":
                _, var_names, expr = stmt
                for var_name in var_names:
                    local_env[var_name] = None  # Type unknown from unpack

        # Check return expression
        body_type = infer_type(body, local_env)
        if return_type and body_type and not types_compatible(return_type, body_type):
            errors.append(
                f"Function '{name}' return type mismatch: declared {type_to_str(return_type)}, "
                f"but body has type {type_to_str(body_type)}"
            )

    def check_class(name, class_def):
        """Check a class definition for type errors."""
        class_params = class_def["class_params"]
        lambda_params = class_def["lambda_params"]
        body = class_def["body"]
        return_type = class_def.get("return_type")
        loss_body = class_def.get("loss_body")
        loss_params = class_def.get("loss_params", [])

        # Register class
        class_env[name] = class_def

        # Build local environment from class params and lambda params
        local_env = {}
        for param_name, param_type in class_params:
            local_env[param_name] = param_type
        for param_name, param_type in lambda_params:
            local_env[param_name] = param_type

        # Check forward body
        body_type = infer_type(body, local_env)
        if return_type and body_type and not types_compatible(return_type, body_type):
            errors.append(
                f"Class '{name}' forward return type mismatch: declared {type_to_str(return_type)}, "
                f"but body has type {type_to_str(body_type)}"
            )

        # Check loss body if present
        if loss_body:
            loss_env = dict(local_env)
            for param_name, param_type in loss_params:
                loss_env[param_name] = param_type
            infer_type(loss_body, loss_env)

    # Check functions
    for name, func_def in unified_ast["functions"].items():
        check_function(name, func_def)

    # Check classes
    for name, class_def in unified_ast["classes"].items():
        check_class(name, class_def)

    # Check program statements
    for stmt in unified_ast["program"]:
        if stmt is None:
            continue
        # Skip func_def and class_def markers
        if stmt[0] in ("func_def", "class_def"):
            continue
        check_statement(stmt)

    return errors
