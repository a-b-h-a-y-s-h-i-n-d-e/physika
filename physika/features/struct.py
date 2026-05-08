import re
from physika.elf import ELF
from typing import Optional
from physika import parser as parser_mod



def is_learnable(type_spec) -> bool:
    """
    Helper functions that returns True for types that should become
    ``nn.Parameter``.
    """
    if type_spec in ("ℝ", "R"):
        return True
    if isinstance(type_spec, tuple) and type_spec[0] == "tensor":
        return True
    return False


def is_submodule(type_spec) -> bool:
    """
    Helper functions that returns True for class instance types.
    """
    return isinstance(type_spec, tuple) and type_spec[0] == "struct_type"


def replace_class_params(code: str, all_params: list) -> str:
    """Rewrite bare param names to ``self.param`` in generated code."""
    for cp_name, _ in all_params:
        code = re.sub(rf'(?<!\.)\b{cp_name}\b', f'self.{cp_name}', code)
    return code


def unwrap_return(ret):
    if ret is None:
        return None
    if ret[0] == "return_single":
        return ret[1]
    if ret[0] == "return_tuple":
        return ("tuple_return", ret[1], ret[2])
    return None



def build_class(constructor_params: Optional[list],
                 body_items: list) -> dict:
    """
    Build a dict of the class from parsed body items.
    """
    body_fields = [(item[1], item[2]) for item in body_items
                   if item[0] == "field_decl"]
    methods = [item[1] for item in body_items if item[0] == "method_def"]

    if constructor_params is None:
        return {"constructor_params": body_fields, "body_fields": [],
                "methods": methods}
    return {"constructor_params": list(constructor_params),
            "body_fields": body_fields, "methods": methods}


def emit_method(method: dict, all_params: list, to_expr,
                 scalar_only: bool) -> list[str]:
    """
    Emit code for a method of an ``nn.Module`` class.
    """
    from physika.utils.ast_utils import emit_body_stmts

    raw_name = method["name"]
    py_name = "forward" if raw_name == "λ" else raw_name
    params = method.get("params", [])
    statements = method.get("statements", [])
    body = method.get("body")

    param_names = [p[0] for p in params]
    lines: list[str] = [
        "",
        f"    def {py_name}(self, {', '.join(param_names)}):",
        "        this = self",
    ]

    for pname, ptype in params:
        if is_learnable(ptype):
            lines.append(f"        {pname} = torch.as_tensor({pname}).float()")

    def apply(code: str) -> str:
        code = re.sub(r'\bthis\b', 'self', code)
        return replace_class_params(code, all_params)

    if statements:
        stmt_lines: list[str] = []
        emit_body_stmts(statements, 2, stmt_lines, list(param_names),
                        set(), to_expr, scalar_only)
        for line in stmt_lines:
            lines.append(apply(line))

    if body is not None:
        if isinstance(body, tuple) and body[0] == "tuple_return":
            _, e1, e2 = body
            lines.append(
                f"        return ({apply(to_expr(e1))}, {apply(to_expr(e2))})"
            )
        else:
            lines.append(f"        return {apply(to_expr(body))}")

    return lines


def generate_class(name: str, class_def: dict) -> str:
    """
    Emit a complete ``nn.Module`` subclass from a class_def.
    """
    from physika.utils.ast_utils import ast_to_torch_expr, ast_uses_func
    from typing import cast

    constructor_params = cast(list, class_def["constructor_params"])
    body_fields = cast(list, class_def.get("body_fields", []))
    methods = cast(list, class_def["methods"])

    all_params = list(constructor_params) + list(body_fields)

    forward = next((m for m in methods if m["name"] == "λ"), None)
    loss = next((m for m in methods if m["name"] == "loss"), None)
    extra = [m for m in methods if m["name"] not in ("λ", "loss")]

    # class header
    lines = [f"class {name}(nn.Module):"]

    # initiailizer
    init_names = [p[0] for p in constructor_params]
    lines.append(f"    def __init__(self, {', '.join(init_names)}):")
    lines.append("        super().__init__()")

    for pname, ptype in constructor_params:
        if is_submodule(ptype):
            lines.append(f"        self.add_module('{pname}', {pname})")
        elif is_learnable(ptype):
            lines.append(
                f"        self.{pname} = "
                f"{pname}.float() if isinstance({pname}, torch.Tensor) "
                f"else nn.Parameter(torch.tensor({pname}).float())"
            )
        else:
            lines.append(f"        self.{pname} = {pname}")

    for fname, ftype in body_fields:
        if isinstance(ftype, tuple) and ftype[0] == "tensor":
            dims = ", ".join(str(d) for d in ftype[1])
            lines.append(
                f"        self.{fname} = nn.Parameter(torch.zeros({dims}))"
            )
        elif is_learnable(ftype):
            lines.append(
                f"        self.{fname} = nn.Parameter(torch.tensor(0.0))"
            )
        else:
            lines.append(f"        self.{fname} = None")

    # forward method (λ)
    if forward:
        scalar_fwd = all(pt == "ℝ" for _, pt in forward.get("params", []))
        lines.extend(
            emit_method(forward, all_params, ast_to_torch_expr, scalar_fwd)
        )

    # loss method
    if loss:
        loss_stmts = loss.get("statements", [])
        loss_body_expr = loss.get("body")
        uses_grad = ast_uses_func(loss_body_expr, "grad") or any(
            ast_uses_func(s, "grad") for s in loss_stmts
        )
        loss_params = list(loss.get("params", []))
        if uses_grad and forward and forward.get("params"):
            fwd_input = forward["params"][0][0]
            loss_params = loss_params + [(fwd_input, "ℝ")]
        lines.extend(
            emit_method({**loss, "params": loss_params},
                         all_params, ast_to_torch_expr, True)
        )

    # extra user-defined methods
    for method in extra:
        lines.extend(
            emit_method(method, all_params, ast_to_torch_expr, False)
        )

    # params property + gradient descent update helper
    lines += [
        "",
        "    @property",
        "    def params(self):",
        "        return list(self.parameters())",
        "",
        "    def update(self, lr, grads):",
        "        with torch.no_grad():",
        "            for p, g in zip(self.parameters(), grads):",
        "                if g is not None:",
        "                    p -= lr * g",
    ]

    return "\n".join(lines)


def make_parser_rules():
    """Return PLY grammar functions for the unified class syntax."""


    def p_statement_class_no_params(p):
        """statement : CLASS ID COLON NEWLINE INDENT class_items DEDENT"""
        name = p[2]
        class_def = build_class(name, None, p[6])
        
        parser_mod.symbol_table[name] = {"type": "class", "value": class_def}
        p[0] = ("class_def", name)

    def p_statement_class_with_params(p):
        """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT class_items DEDENT"""
        name = p[2]
        class_def = build_class(name, p[4], p[9])

        parser_mod.symbol_table[name] = {"type": "class", "value": class_def}
        p[0] = ("class_def", name)

    def p_class_items_multi(p):
        """class_items : class_items class_item"""
        p[0] = p[1] + [p[2]]

    def p_class_items_single(p):
        """class_items : class_item"""
        p[0] = [p[1]]

    def p_class_item_field(p):
        """class_item : ID COLON type_spec NEWLINE"""
        p[0] = ("field_decl", p[1], p[3])

    def p_class_item_method(p):
        """class_item : class_method"""
        p[0] = ("method_def", p[1])

    # arrow syntax
    def p_class_method_arrow_params_body(p):
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": p[7],
                "statements": p[11], "body": unwrap_return(p[12])}

    def p_class_method_arrow_params_simple(p):
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": p[7],
                "statements": [], "body": unwrap_return(p[11])}

    def p_class_method_arrow_no_params_body(p):
        """class_method : DEF ID LPAREN RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": [], "return_type": p[6],
                "statements": p[10], "body": unwrap_return(p[11])}

    def p_class_method_arrow_no_params_simple(p):
        """class_method : DEF ID LPAREN RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": [], "return_type": p[6],
                "statements": [], "body": unwrap_return(p[10])}

    def p_class_method_colon_params_body(p):
        """class_method : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": p[7],
                "statements": p[11], "body": unwrap_return(p[12])}

    def p_class_method_colon_params_simple(p):
        """class_method : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": p[7],
                "statements": [], "body": unwrap_return(p[11])}

    def p_class_method_colon_no_params_body(p):
        """class_method : DEF ID LPAREN RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": [], "return_type": p[6],
                "statements": p[10], "body": unwrap_return(p[11])}

    def p_class_method_colon_no_params_simple(p):
        """class_method : DEF ID LPAREN RPAREN COLON type_spec COLON NEWLINE INDENT class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": [], "return_type": p[6],
                "statements": [], "body": unwrap_return(p[10])}

    def p_class_method_no_ret_body(p):
        """class_method : DEF ID LPAREN params RPAREN COLON NEWLINE INDENT func_body_stmts DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": None,
                "statements": p[9], "body": None}

    def p_class_method_no_ret_simple(p):
        """class_method : DEF ID LPAREN params RPAREN COLON NEWLINE INDENT class_method_return DEDENT"""
        p[0] = {"name": p[2], "params": p[4], "return_type": None,
                "statements": [], "body": unwrap_return(p[9])}

    def p_class_method_return_single(p):
        """class_method_return : RETURN func_expr NEWLINE"""
        p[0] = ("return_single", p[2])

    def p_class_method_return_tuple(p):
        """class_method_return : RETURN func_expr COMMA func_expr NEWLINE"""
        p[0] = ("return_tuple", p[2], p[4])

    def p_factor_field(p):
        """factor : factor DOT ID"""
        p[0] = ("field_access", p[1], p[3])

    def p_func_factor_field(p):
        """func_factor : func_factor DOT ID"""
        p[0] = ("field_access", p[1], p[3])

    def p_factor_method_call(p):
        """factor : factor DOT ID LPAREN args RPAREN"""
        p[0] = ("method_call", p[1], p[3], p[5] or [])

    def p_func_factor_method_call(p):
        """func_factor : func_factor DOT ID LPAREN func_args RPAREN"""
        p[0] = ("method_call", p[1], p[3], p[5] or [])

    def p_type_class(p):
        """type_spec : ID"""
        p[0] = ("struct_type", p[1])

    def p_func_body_stmt_this_assign(p):
        """func_body_stmt : ID DOT ID EQUALS func_expr NEWLINE"""
        p[0] = ("body_this_assign", p[1], p[3], p[5])

    def p_func_body_stmt_this_tuple_assign(p):
        """func_body_stmt : ID DOT ID COMMA ID DOT ID EQUALS func_expr NEWLINE"""
        p[0] = ("body_this_tuple_unpack", [(p[1], p[3]), (p[5], p[7])], p[9])

    def p_func_body_stmt_method_call(p):
        """func_body_stmt : func_factor DOT ID LPAREN func_args RPAREN NEWLINE"""
        p[0] = ("body_expr", ("method_call", p[1], p[3], p[5] or []))

    return [
        p_statement_class_no_params,
        p_statement_class_with_params,
        p_class_items_multi,
        p_class_items_single,
        p_class_item_field,
        p_class_item_method,
        p_class_method_arrow_params_body,
        p_class_method_arrow_params_simple,
        p_class_method_arrow_no_params_body,
        p_class_method_arrow_no_params_simple,
        p_class_method_colon_params_body,
        p_class_method_colon_params_simple,
        p_class_method_colon_no_params_body,
        p_class_method_colon_no_params_simple,
        p_class_method_no_ret_body,
        p_class_method_no_ret_simple,
        p_class_method_return_single,
        p_class_method_return_tuple,
        p_factor_field,
        p_func_factor_field,
        p_factor_method_call,
        p_func_factor_method_call,
        p_type_class,
        p_func_body_stmt_this_assign,
        p_func_body_stmt_this_tuple_assign,
        p_func_body_stmt_method_call,
    ]




class StructFeature(ELF):
    name = "struct"

    def lexer_rules(self) -> dict:
        def t_DOT(t):
            r"\."
            return t
        return {"tokens": ["DOT"], "token_funcs": [t_DOT]}

    def parser_rules(self) -> list:
        return make_parser_rules()

    def type_rules(self) -> dict:
        from physika.utils.types import TInstance
        from physika.utils.type_checker_utils import from_typespec, type_to_str

        def check_not_constructor(obj_expr, class_env, add_error, what):
            if (isinstance(obj_expr, tuple) and obj_expr[0] == "var"
                    and obj_expr[1] in class_env):
                add_error(
                    f"'{obj_expr[1]}' is a class constructor, not an instance; "
                    f"use an instance to access {what}"
                )
                return True
            return False

        def check_field_access(node, env, s, func_env, class_env,
                               add_error, infer_expr):
            _, obj_expr, field_name = node
            obj_type, s = infer_expr(obj_expr, env, s, func_env, class_env,
                                     add_error)
            from physika.utils.types import TInstance
            if isinstance(obj_type, TInstance):
                info = class_env.get(obj_type.class_name)
                if info:
                    fields = dict(info.get("fields", []))
                    if field_name in fields:
                        from physika.utils.type_checker_utils import from_typespec
                        return from_typespec(fields[field_name]), s
                    add_error(
                        f"Class '{obj_type.class_name}' has no field '{field_name}'"
                    )
            elif obj_type is None:
                check_not_constructor(obj_expr, class_env, add_error,
                                       f"field '{field_name}'")
            return None, s

        def check_method_call(node, env, s, func_env, class_env,
                              add_error, infer_expr):
            _, obj_expr, method_name, args = node
            obj_type, s = infer_expr(obj_expr, env, s, func_env, class_env,
                                     add_error)
            
            if obj_type is None:
                check_not_constructor(obj_expr, class_env, add_error,
                                       f"method '{method_name}'")
                return None, s
            if isinstance(obj_type, TInstance):
                info = class_env.get(obj_type.class_name)
                if info:
                    methods = info.get("methods", {})
                    if method_name in methods:
                        method_info = methods[method_name]
                        expected_params = method_info.get("params", [])
                        if len(args) != len(expected_params):
                            add_error(
                                f"Method '{obj_type.class_name}.{method_name}' expects "
                                f"{len(expected_params)} argument(s), got {len(args)}"
                            )
                        else:
                            for arg, (pname, ptype_spec) in zip(args, expected_params):
                                arg_type, s = infer_expr(arg, env, s, func_env,
                                                         class_env, add_error)
                                expected_hm = from_typespec(ptype_spec)
                                if arg_type is None or expected_hm is None:
                                    continue

                                exp_is_inst = isinstance(expected_hm, TInstance)
                                arg_is_inst = isinstance(arg_type, TInstance)
                                if exp_is_inst and not arg_is_inst:
                                    add_error(
                                        f"Method '{obj_type.class_name}.{method_name}' "
                                        f"parameter '{pname}': expected "
                                        f"'{expected_hm.class_name}', "
                                        f"got '{type_to_str(arg_type)}'"
                                    )
                                elif exp_is_inst and arg_is_inst and expected_hm.class_name != arg_type.class_name:
                                    add_error(
                                        f"Method '{obj_type.class_name}.{method_name}' "
                                        f"parameter '{pname}': expected "
                                        f"'{expected_hm.class_name}', "
                                        f"got '{arg_type.class_name}'"
                                    )
                                elif not exp_is_inst and arg_is_inst:
                                    add_error(
                                        f"Method '{obj_type.class_name}.{method_name}' "
                                        f"parameter '{pname}': expected "
                                        f"'{type_to_str(expected_hm)}', "
                                        f"got '{arg_type.class_name}'"
                                    )
                        return from_typespec(method_info.get("return_type")), s
                    add_error(
                        f"Class '{obj_type.class_name}' has no method '{method_name}'"
                    )
            return None, s

        return {
            "field_access": check_field_access,
            "method_call":  check_method_call,
        }

    def forward_rules(self) -> dict:
        def emit_field_access(node, to_expr, **ctx):
            _, obj_expr, field_name = node
            return f"{to_expr(obj_expr)}.{field_name}"

        def emit_method_call(node, to_expr, **ctx):
            _, obj_expr, method_name, args = node
            args_str = ", ".join(to_expr(a) for a in args)
            return f"{to_expr(obj_expr)}.{method_name}({args_str})"

        def emit_class_def(node, to_expr, **ctx):
            _, name, class_def = node
            return generate_class(name, class_def)

        return {
            "field_access": emit_field_access,
            "method_call":  emit_method_call,
            "class_def":    emit_class_def,
        }
