import ply.yacc as yacc

from lexer import tokens, lexer

from utils import is_torch_tensor, from_torch

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


symbol_table = {}
print_separator = False
parsing_function_body = False

# Collect all statements as AST, evaluate after parsing
program_ast = []  # List of AST statements
AST_MODE = True   # When True, build AST instead of immediate evaluation

# Flag to enable AST printing (set via --ast flag)
PRINT_AST = False


# AST PRINTING HELPER FUNCTIONS
def print_ast(node, indent=0):
    """Pretty-print an AST node"""
    prefix = "  " * indent
    if node is None:
        print(f"{prefix}None")
    elif isinstance(node, tuple):
        print(f"{prefix}({node[0]}")
        for child in node[1:]:
            print_ast(child, indent + 1)
        print(f"{prefix})")
    elif isinstance(node, list):
        print(f"{prefix}[")
        for item in node:
            print_ast(item, indent + 1)
        print(f"{prefix}]")
    elif isinstance(node, dict):
        print(f"{prefix}{{")
        for k, v in node.items():
            print(f"{prefix}  {k}:")
            print_ast(v, indent + 2)
        print(f"{prefix}}}")
    else:
        print(f"{prefix}{repr(node)}")

def print_symbol_ast(name, entry):
    """Print the AST for a symbol table entry"""
    print(f"\n--- {name} ---")
    if entry["type"] == "function":
        print(f"Type: function")
        print(f"Params: {entry['value']['params']}")
        print(f"Return type: {entry['value']['return_type']}")
        print(f"Body AST:")
        print_ast(entry['value']['body'], 1)
    elif entry["type"] == "class":
        print(f"Type: class")
        print(f"Class params: {entry['value']['class_params']}")
        print(f"Lambda params: {entry['value']['lambda_params']}")
        print(f"Return type: {entry['value']['return_type']}")
        if entry['value'].get('has_loop'):
            print(f"Loop var: {entry['value'].get('loop_var')}")
            print(f"Loop body AST:")
            print_ast(entry['value'].get('loop_body'), 1)
        print(f"Body AST:")
        print_ast(entry['value']['body'], 1)
        if entry['value'].get('has_loss'):
            print(f"Loss params: {entry['value'].get('loss_params')}")
            print(f"Loss body AST:")
            print_ast(entry['value'].get('loss_body'), 1)
    else:
        print(f"Type: {entry['type']}")
        if 'value' in entry:
            val = entry['value']
            # Don't print large tensor values
            if hasattr(val, 'shape'):
                print(f"Value: <tensor shape={val.shape}>")
            elif isinstance(val, list) and len(str(val)) > 100:
                print(f"Value: <list len={len(val)}>")
            else:
                print(f"Value: {val}")


# PARSER
precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("left", "MATMUL"),
    ("right", "POWER"),
    ("right", "EQUALS"),
)


# Program
def p_program(p):
    """program : statements"""
    p[0] = p[1]

def p_statements_multi(p):
    """statements : statements statement"""
    p[0] = p[1] + ([] if p[2] is None else [p[2]])

def p_statements_single(p):
    """statements : statement"""
    p[0] = [] if p[1] is None else [p[1]]


# Types
def p_type_scalar(p):
    """type_spec : TYPE"""
    if p[1] == "ℤ":
        p[0] = "ℤ"
    elif p[1] == "ℕ":
        p[0] = "ℕ"
    else:
        p[0] = "ℝ"

def p_type_function(p):
    """type_spec : TYPE ARROW TYPE"""
    # Function type: R → R
    p[0] = ("func_type", p[1], p[3])

def p_type_tangent(p):
    """type_spec : TANGENT ID TYPE"""
    # T_x M notation for tangent space at point x of manifold M
    p[0] = ("tangent", p[2], p[3])

def p_type_tensor(p):
    """type_spec : TYPE LBRACKET dimension_list RBRACKET"""
    p[0] = ("tensor", p[3])

def p_dimension_list_single(p):
    """dimension_list : dimension_spec"""
    p[0] = [p[1]]

def p_dimension_list_multi(p):
    """dimension_list : dimension_spec COMMA dimension_list"""
    p[0] = [p[1]] + p[3]

def p_dimension_contravariant(p):
    """dimension_spec : PLUS NUMBER"""
    p[0] = (int(p[2]), "contravariant")

def p_dimension_covariant(p):
    """dimension_spec : MINUS NUMBER"""
    p[0] = (int(p[2]), "covariant")

def p_dimension_invariant(p):
    """dimension_spec : NUMBER"""
    p[0] = (int(p[1]), "invariant")


# Statements
def p_statement_function(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT"""
    name, params, return_type, body = p[2], p[4], p[7], p[12]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "statements": []
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)

def p_statement_function_with_body(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT"""
    # def funcname(params): return_type:
    #     stmt1
    #     stmt2
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    body_stmts = p[11]
    final_expr = p[13]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body": final_expr,
        "has_loop": False,
        "statements": body_stmts
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)

def p_func_body_stmts_single(p):
    """func_body_stmts : func_body_stmt"""
    p[0] = [p[1]] if p[1] else []

def p_func_body_stmts_multi(p):
    """func_body_stmts : func_body_stmts func_body_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])

def p_func_body_stmt_assign(p):
    """func_body_stmt : ID EQUALS func_expr NEWLINE"""
    # Simple assignment: x = expr
    p[0] = ("body_assign", p[1], p[3])

def p_func_body_stmt_decl(p):
    """func_body_stmt : ID COLON type_spec EQUALS func_expr NEWLINE"""
    # Typed declaration: x : R = expr
    p[0] = ("body_decl", p[1], p[3], p[5])

def p_func_body_stmt_tuple_unpack(p):
    """func_body_stmt : ID COMMA ID EQUALS func_expr NEWLINE"""
    # Tuple unpacking: a, b = expr
    p[0] = ("body_tuple_unpack", [p[1], p[3]], p[5])

def p_func_body_stmt_tuple_unpack_three(p):
    """func_body_stmt : ID COMMA ID COMMA ID EQUALS func_expr NEWLINE"""
    # Tuple unpacking: a, b, c = expr
    p[0] = ("body_tuple_unpack", [p[1], p[3], p[5]], p[7])

def p_func_body_stmt_empty(p):
    """func_body_stmt : NEWLINE"""
    p[0] = None

def p_statement_function_with_loop(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_init FOR ID COLON NEWLINE INDENT func_loop_body DEDENT NEWLINE RETURN func_expr DEDENT"""
    # def funcname(params): return_type:
    #     init_stmts
    #     for i:
    #         loop_body
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    init_stmts = p[11]   # After first INDENT
    loop_var = p[13]     # FOR ID
    loop_body = p[17]    # After second INDENT
    final_expr = p[21]   # After RETURN

    # Find arrays indexed by loop var to determine iteration count at runtime
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    func_def = {
        "params": params,
        "return_type": return_type,
        "has_loop": True,
        "init_stmts": init_stmts,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,  # Store for runtime inference
        "loop_body": loop_body,
        "body": final_expr
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)

def p_func_init_empty(p):
    """func_init : """
    p[0] = []

def p_func_init_multi(p):
    """func_init : func_init func_init_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])

def p_func_init_stmt_assign(p):
    """func_init_stmt : ID EQUALS func_expr NEWLINE"""
    p[0] = ("init_assign", p[1], p[3])

def p_func_init_stmt_empty(p):
    """func_init_stmt : NEWLINE"""
    p[0] = None

def p_func_loop_body_empty(p):
    """func_loop_body : """
    p[0] = []

def p_func_loop_body_multi(p):
    """func_loop_body : func_loop_body func_loop_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])

def p_func_loop_stmt_assign(p):
    """func_loop_stmt : ID EQUALS func_expr NEWLINE"""
    p[0] = ("loop_assign", p[1], p[3])

def p_func_loop_stmt_pluseq(p):
    """func_loop_stmt : ID PLUSEQ func_expr NEWLINE"""
    p[0] = ("loop_pluseq", p[1], p[3])

def p_func_loop_stmt_empty(p):
    """func_loop_stmt : NEWLINE"""
    p[0] = None

def p_statement_class(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return expr
    class_name = p[2]
    class_params = p[4]  # Parameters for the class (like weights)
    lambda_params = p[12]  # Parameters for the lambda (like input x)
    return_type = p[15]
    body = p[20]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "has_loss": False
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)

def p_statement_class_with_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return forward_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body = p[20]
    loss_name = p[24]  # ID after DEF
    loss_params = p[26]
    loss_return_type = p[29]
    loss_body = p[34]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "has_loss": True,
        "loss_params": loss_params,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)

def p_statement_class_with_loop(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT FOR ID COLON NEWLINE INDENT lambda_loop_body DEDENT RETURN func_expr NEWLINE DEDENT DEDENT"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i:
    #             x = expr
    #         return final_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]   # After first INDENT
    return_type = p[15]
    loop_var = p[20]        # FOR ID
    loop_body = p[24]       # After third INDENT
    final_expr = p[27]      # After DEDENT RETURN

    # Find arrays indexed by loop var
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": False,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,
        "loop_body": loop_body,
        "body": final_expr
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)

def p_statement_class_with_loop_and_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT FOR ID COLON NEWLINE INDENT lambda_loop_body DEDENT RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i:
    #             x = expr
    #         return final_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    loop_var = p[20]
    loop_body = p[24]
    final_expr = p[27]      # After DEDENT RETURN
    loss_name = p[31]       # ID after DEF
    loss_params = p[33]     # params after LPAREN
    loss_return_type = p[36]
    loss_body = p[41]

    # Find arrays indexed by loop var
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": True,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,
        "loop_body": loop_body,
        "body": final_expr,
        "loss_params": loss_params,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)

def p_lambda_loop_body_empty(p):
    """lambda_loop_body : """
    p[0] = []

def p_lambda_loop_body_multi(p):
    """lambda_loop_body : lambda_loop_body lambda_loop_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])

def p_lambda_loop_stmt_assign(p):
    """lambda_loop_stmt : ID EQUALS func_expr NEWLINE"""
    p[0] = ("loop_assign", p[1], p[3])

def p_lambda_loop_stmt_empty(p):
    """lambda_loop_stmt : NEWLINE"""
    p[0] = None

def p_statement_decl(p):
    """statement : ID COLON type_spec EQUALS expr"""
    name = p[1]
    type_spec = p[3]
    expr_ast = p[5]  # This is now an AST, not an evaluated value

    # Return AST node for declaration (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("decl", name, type_spec, expr_ast, p.lineno(1))

def p_statement_assign(p):
    """statement : ID EQUALS expr"""
    name = p[1]
    expr_ast = p[3]  # This is now an AST

    # Return AST node for assignment (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("assign", name, expr_ast, p.lineno(1))

def p_statement_expr(p):
    """statement : expr"""
    # Return AST node for standalone expression (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("expr", p[1], p.lineno(1))

def p_statement_empty(p):
    """statement : NEWLINE"""
    global print_separator
    if p[1] >= 2:
        print_separator = True
    p[0] = None


# Helpers
def find_indexed_arrays(ast, loop_var):
    """Find array names that are indexed by the loop variable in an AST."""
    arrays = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, tuple):
            # Check for array index pattern: ("index", array_name, index_var)
            if len(node) >= 3 and node[0] == "index":
                array_name = node[1]
                index_expr = node[2]
                # Check if index is the loop variable
                # Handle various AST representations of the index
                is_loop_var = (
                    index_expr == ("var", loop_var) or
                    index_expr == loop_var or
                    (loop_var == "i" and index_expr == ("imaginary",))  # 'i' can be IMAGINARY token
                )
                if is_loop_var:
                    if isinstance(array_name, str):
                        arrays.append(array_name)
                    elif isinstance(array_name, tuple) and array_name[0] == "var":
                        arrays.append(array_name[1])
            # Recurse into tuple elements
            for item in node:
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return arrays

def p_statement_for(p):
    """statement : FOR ID COLON NEWLINE INDENT for_body DEDENT"""
    loop_var = p[2]
    body_statements = p[6]

    # Find arrays indexed by loop variable to determine iteration count at eval time
    indexed_arrays = []
    for stmt in body_statements:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    # Return AST node - iteration count will be determined during evaluation
    # Include line number for error reporting
    p[0] = ("for_loop", loop_var, body_statements, indexed_arrays, p.lineno(1))

def p_for_body_empty(p):
    """for_body : """
    p[0] = []

def p_for_body_multi(p):
    """for_body : for_body for_statement"""
    p[0] = p[1] + ([p[2]] if p[2] is not None else [])

def p_for_statement_assign(p):
    """for_statement : ID EQUALS func_expr NEWLINE"""
    # Store as AST to be evaluated later
    p[0] = ("for_assign", p[1], p[3])

def p_for_statement_pluseq(p):
    """for_statement : ID PLUSEQ func_expr NEWLINE"""
    # Store as AST: x += expr becomes x = x + expr
    p[0] = ("for_pluseq", p[1], p[3])

def p_for_statement_call(p):
    """for_statement : ID LPAREN func_args RPAREN NEWLINE"""
    # Store function call as AST
    p[0] = ("for_call", p[1], p[3])

def p_for_statement_empty(p):
    """for_statement : NEWLINE"""
    p[0] = None

def execute_for_statement(stmt):
    """Execute a statement inside a for loop"""
    if stmt is None:
        return

    op = stmt[0]

    if op == "for_assign":
        _, name, ast_expr = stmt
        # Evaluate the AST expression with current symbol table
        value = evaluate_ast(ast_expr, {})
        if name in symbol_table:
            symbol_table[name]["value"] = value
        else:
            symbol_table[name] = {"type": "ℝ", "value": value}

    elif op == "for_pluseq":
        _, name, ast_expr = stmt
        # x += expr means x = x + expr
        rhs_value = evaluate_ast(ast_expr, {})
        if name in symbol_table:
            current = symbol_table[name]["value"]
            symbol_table[name]["value"] = current + rhs_value
        else:
            raise NameError(f"Variable '{name}' not defined for +=")

    elif op == "for_call":
        _, func_name, arg_asts = stmt
        # Evaluate arguments
        args = [evaluate_ast(arg, {}) for arg in arg_asts]
        execute_for_call(func_name, args)

def execute_for_call(func_name, args):
    """Execute a function call inside a for loop"""
    if func_name == "print":
        if len(args) == 1:
            value = args[0]
            display_value = from_torch(value) if is_torch_tensor(value) else value
            print(f"  {display_value}")


# Parameters
def p_params_empty(p):
    """params : """
    p[0] = []

def p_params_single(p):
    """params : ID COLON type_spec"""
    p[0] = [(p[1], p[3])]

def p_params_multi(p):
    """params : ID COLON type_spec COMMA params"""
    p[0] = [(p[1], p[3])] + p[5]

# Function Expression (AST nodes)
def p_func_expr_plus(p):
    """func_expr : func_expr PLUS func_term"""
    p[0] = ("add", p[1], p[3])

def p_func_expr_minus(p):
    """func_expr : func_expr MINUS func_term"""
    p[0] = ("sub", p[1], p[3])

def p_func_expr_term(p):
    """func_expr : func_term"""
    p[0] = p[1]

def p_func_term_times(p):
    """func_term : func_term TIMES func_factor
                 | func_term DIVIDE func_factor
                 | func_term MATMUL func_factor
                 | func_term POWER func_factor"""
    if p[2] == "*":
        p[0] = ("mul", p[1], p[3])
    elif p[2] == "/":
        p[0] = ("div", p[1], p[3])
    elif p[2] == "**":
        p[0] = ("pow", p[1], p[3])
    else:  # @
        p[0] = ("matmul", p[1], p[3])

def p_func_term_factor(p):
    """func_term : func_factor"""
    p[0] = p[1]

def p_func_factor_number(p):
    """func_factor : NUMBER"""
    p[0] = ("num", p[1])

def p_func_factor_id(p):
    """func_factor : ID"""
    p[0] = ("var", p[1])

def p_func_factor_group(p):
    """func_factor : LPAREN func_expr RPAREN"""
    p[0] = p[2]

def p_func_factor_call(p):
    """func_factor : ID LPAREN func_args RPAREN"""
    p[0] = ("call", p[1], p[3])

def p_func_factor_index(p):
    """func_factor : ID LBRACKET func_expr RBRACKET"""
    # Tensor indexing: W[i]
    p[0] = ("index", p[1], p[3])

def p_func_factor_call_index(p):
    """func_factor : ID LPAREN func_args RPAREN LBRACKET func_expr RBRACKET"""
    # Indexing a function call result: grad(H, x)[0]
    p[0] = ("call_index", p[1], p[3], p[6])

def p_func_factor_array(p):
    """func_factor : LBRACKET func_elements RBRACKET"""
    # Array literal in function body: [1.0, 2.0, 3.0]
    p[0] = ("array", p[2])

def p_func_factor_string(p):
    """func_factor : STRING"""
    # String literal (for equations): 'x0 = a + b'
    p[0] = ("string", p[1])

def p_func_factor_imaginary(p):
    """func_factor : IMAGINARY"""
    # Imaginary unit i
    p[0] = ("imaginary",)

def p_func_elements_single(p):
    """func_elements : func_expr"""
    p[0] = [p[1]]

def p_func_elements_multi(p):
    """func_elements : func_expr COMMA func_elements"""
    p[0] = [p[1]] + p[3]

def p_func_args_empty(p):
    """func_args : """
    p[0] = []

def p_func_args_single(p):
    """func_args : func_expr"""
    p[0] = [p[1]]

def p_func_args_multi(p):
    """func_args : func_expr COMMA func_args"""
    p[0] = [p[1]] + p[3]

# Expressions (evaluated immediately)
def execute_statement(stmt):
    """Execute a statement tuple from for loop body"""
    if stmt[0] == "assign":
        _, name, value = stmt
        if name in symbol_table:
            symbol_table[name]["value"] = value
        else:
            symbol_table[name] = {"type": "ℝ", "value": value}
    elif stmt[0] == "decl":
        _, name, type_spec, value = stmt
        symbol_table[name] = {"type": type_spec, "value": value}

def create_pytorch_model(class_def):
    """Create a PyTorch nn.Module from class definition"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")

    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()

            # Build layers from class body
            for stmt in class_def["body"]:
                if stmt[0] == "layer":
                    layer_name, layer_spec = stmt[1], stmt[2]
                    self.layers[layer_name] = build_layer(layer_spec)

        def forward(self, x):
            # Execute forward pass through sequential layers
            for layer in self.layers.values():
                x = layer(x)
            return x

    return DynamicModel()

def build_layer(layer_spec):
    """Build a PyTorch layer from specification"""
    layer_type = layer_spec[0]

    if layer_type == "linear":
        in_features, out_features = layer_spec[1], layer_spec[2]
        return nn.Linear(in_features, out_features)

    elif layer_type == "sequential":
        layers = [build_layer(spec) for spec in layer_spec[1]]
        return nn.Sequential(*layers)

    elif layer_type == "tanh":
        return nn.Tanh()

    elif layer_type == "relu":
        return nn.ReLU()

    elif layer_type == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unknown layer type: {layer_type}")

def p_expr_plus(p):
    """expr : expr PLUS term"""
    # Build AST node instead of immediate evaluation
    p[0] = ("add", p[1], p[3])

def p_expr_minus(p):
    """expr : expr MINUS term"""
    p[0] = ("sub", p[1], p[3])

def p_expr_term(p):
    """expr : term"""
    p[0] = p[1]

def p_term_binop(p):
    """term : term TIMES factor
            | term DIVIDE factor
            | term MATMUL factor
            | term POWER factor"""
    if p[2] == "*":
        p[0] = ("mul", p[1], p[3])
    elif p[2] == "/":
        p[0] = ("div", p[1], p[3])
    elif p[2] == "**":
        p[0] = ("pow", p[1], p[3])
    else:  # @
        p[0] = ("matmul", p[1], p[3])

# Factors
def p_term_factor(p):
    """term : factor"""
    p[0] = p[1]


def p_factor_call(p):
    """factor : ID LPAREN args RPAREN"""
    # Return AST node for function call
    func_name = p[1]
    args = p[3]
    p[0] = ("call", func_name, args)

def p_factor_number(p):
    """factor : NUMBER"""
    # Return AST node for number literal
    p[0] = ("num", p[1])

def p_factor_neg(p):
    """factor : MINUS factor"""
    # Return AST node for unary minus
    p[0] = ("neg", p[2])

def p_factor_id(p):
    """factor : ID"""
    # Return AST node for variable reference
    p[0] = ("var", p[1])

def p_factor_group(p):
    """factor : LPAREN expr RPAREN"""
    p[0] = p[2]

def p_factor_array(p):
    """factor : LBRACKET elements RBRACKET"""
    # Return AST node for array literal
    p[0] = ("array", p[2])

def p_factor_string(p):
    """factor : STRING"""
    # String literal (for equations and symbolic): 'x0 = a + b'
    p[0] = ("equation_string", p[1])

def p_elements_single(p):
    """elements : expr"""
    p[0] = [p[1]]

def p_elements_multi(p):
    """elements : expr COMMA elements"""
    p[0] = [p[1]] + p[3]

def p_elements_newline(p):
    """elements : NEWLINE elements
                | elements NEWLINE"""
    # Allow newlines within array definitions
    if len(p) == 3:
        p[0] = p[2] if isinstance(p[2], list) else p[1]

def p_factor_index(p):
    """factor : ID LBRACKET NUMBER RBRACKET"""
    # Return AST node for array indexing
    p[0] = ("index", p[1], ("num", p[3]))

def p_factor_slice(p):
    """factor : ID LBRACKET NUMBER COLON NUMBER RBRACKET"""
    # Return AST node for array slicing
    p[0] = ("slice", p[1], ("num", p[3]), ("num", p[5]))


# Function Arguments
def p_args_empty(p):
    """args : """
    p[0] = []

def p_args_single(p):
    """args : expr"""
    p[0] = [p[1]]

def p_args_multi(p):
    """args : expr COMMA args"""
    p[0] = [p[1]] + p[3]

def p_error(p):
    if p:
        raise SyntaxError(f"Syntax error at '{p.value}'")
    else:
        raise SyntaxError("Syntax error at EOF")


parser = yacc.yacc()


# Only used by the legacy execute_for_statement path
# which is not called in the main pipeline (AST → type_check → codegen → exec).
def evaluate_ast(node, local_scope):
    raise NotImplementedError("evaluate_ast is legacy code; use codegen pipeline instead")
