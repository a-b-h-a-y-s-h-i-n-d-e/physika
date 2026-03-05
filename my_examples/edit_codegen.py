for func_def in unified_ast["functions"].values():
        if ast_uses_func(func_def.get("body"), "grad"):
            needs_grad = True
            break
        if any(ast_uses_func(s, "grad") for s in func_def.get("statements", [])):
            needs_grad = True
            break