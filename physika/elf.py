class ELF:
    """
    Easy Language Feature (ELF) class for implementing Physika language features.

    Subclasses of ``ELF`` extends Physika language features with new parser/lexer,
    type checking, and code generation rules.  Each subclass
    declares a unique `name` and overrides four methods that controls basics operations
    in Physika. These are:

    * `parser_rules`: PLY grammar functions that introduce new syntax rules.
    * `lexer_rules`: New reserved keywords and token names for the lexer.
    * `type_rules`: Physika's type checker handlers that
      receive AST nodes and update the type environment following Hindley-Milner type inference.
    * `forward_rules`: Code generation handlers that emit
      PyTorch code as strings from AST nodes.
    * `backward_rules`: Differentiation handlers for custom
      gradient computation.

    Attributes
    ----------
    name : str
        Identifier language feature.

    Examples
    --------
    >>> from physika.elf import ELF
    >>> from physika.utils.type_checker_utils import Substitution
    >>> class WhileLoopFeature(ELF):
    ...     name = "while_loop"
    ...     def lexer_rules(self):
    ...         return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"]}

    ...     def parser_rules(self):
    ...         def p_while(p):
    ...             \"\"\"statement : WHILE condition COLON NEWLINE INDENT statements DEDENT\"\"\"
    ...             p[0] = ("while_loop", p[2], p[6])
    ...         return [p_while]

    ...     def type_rules(self):
    ...         def check(node, env, s, func_env, class_env, add_error, infer_expr):
    ...             from physika.utils.infer_expr import infer_expr
    ...             _, cond, body = node
    ...             # checking contion is scalar for simplicity
    ...             cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
    ...             if cond_t != ("scalar",):
    ...                 add_error("while condition must be scalar")
    ...             return None, s
    ...         return {"while_loop": check}

    ...     def forward_rules(self):
    ...         def emit(node):
    ...             from physika.utils.ast_utils import generate_statement, condition_to_expr
    ...             _, cond, body = node
    ...             body_lines = [f"    {generate_statement(s, set())}" for s in body]
    ...             body_code = "\\n".join(body_lines) if body_lines else "    pass"
    ...             return f"while {condition_to_expr(cond)}:\\n{body_code}"
    ...         return {"while_loop": emit}

    ...     def backward_rules(self):
    ...         def grad(node, grad_output):
    ...             return (
    ...                 f"_adj = {grad_output}\\n"
    ...                 f"for _state in reversed(_while_tape):\\n"
    ...                 f"    _adj = _body_vjp(_state, _adj)"
    ...             )
    ...         return {"while_loop": grad}
    >>> feature_name = WhileLoopFeature().name
    >>> feature_name
    'while_loop'
    >>> # check parser rules
    >>> parser_rules = WhileLoopFeature().parser_rules()
    >>> parser_rules[0].__doc__
    'statement : WHILE condition COLON NEWLINE INDENT statements DEDENT'
    >>> # check lexer rules
    >>> lexer_rules = WhileLoopFeature().lexer_rules()
    >>> lexer_rules["reserved"]
    {'while': 'WHILE'}
    >>> # check forward emit
    >>> while_forward_emit = WhileLoopFeature().forward_rules()[feature_name]
    >>> node = ("while_loop", ("cond_lt", ("var", "n"), ("num", 10.0)), [("assign", "n", ("add", ("var", "n"), ("num", 1.0)), 1)])
    >>> while_forward_emit(node)
    'while n < 10.0:\\n    n = (n + 1.0)'
    >>> # check backward emit
    >>> while_backward_grad = WhileLoopFeature().backward_rules()[feature_name]
    >>> print(while_backward_grad(node, "dL_dn"))
    _adj = dL_dn
    for _state in reversed(_while_tape):
        _adj = _body_vjp(_state, _adj)
    >>> # check type rules
    >>> check = WhileLoopFeature().type_rules()[feature_name]
    >>> errors = []
    >>> t, _ = check(("while_loop", ("var", "n"), []), {"n": ("scalar",)}, Substitution(), {}, {}, errors.append, None)  # noqa: E501
    >>> t is None and errors == []
    True
    """

    name: str = ""

    def parser_rules(self) -> list:
        """
        Return PLY grammar functions that define new syntax for this feature.

        Each returned callable must be a PLY ``p_`` function. PLY's functions
        must start with ``p_`` and docstring must contain a valid
        Backus-Naur Form (BNF) rule.  These functions are passed to ``parser.py``  # noqa: E501
        by `FeatureRegistry.add_parser_rules` so that
        ``yacc.yacc()`` include them when parsing Physika code.

        Returns
        -------
        list[Callable]
            A list of PLY grammar functions.

        Examples
        --------
        >>> import physika.parser as physika_parser
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def parser_rules(self):
        ...         def p_while(p):
        ...             \"\"\"statement : WHILE expr COLON NEWLINE INDENT stmts DEDENT\"\"\"  # noqa: E501
        ...             p[0] = ("while_loop", p[2], p[6])
        ...         return [p_while]

        >>> WhileLoopFeature().parser_rules()[0].__doc__
        'statement : WHILE expr COLON NEWLINE INDENT stmts DEDENT'
        """
        return []

    def lexer_rules(self) -> dict:
        """
        Adds reserve keywords and token names to lexer environment for
        the new feature.

        The returned dict contains two optional keys:

        - ``"reserved"``: maps keyword strings to PLY token names
          and merges into ``lexer.reserved``
          so the existing ``t_ID`` rule promotes matching identifiers to
          the new token type automatically.
        - ``"tokens"``: list of new token names that will be appended to
          ``lexer.tokens``.


        Returns
        -------
        dict
            Dict with optional keys ``"reserved"`` and ``"tokens"``.

        Examples
        --------
        >>> from physika.elf import ELF
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def lexer_rules(self):
        ...         return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"]}  # noqa: E501
        >>> WhileLoopFeature().lexer_rules()
        {'reserved': {'while': 'WHILE'}, 'tokens': ['WHILE']}
        """
        return {"reserved": {}, "tokens": [], "token_funcs": []}

    def type_rules(self) -> dict:
        """Return a mapping from AST node tags to type inference function handlers.

        Each handler receives the full AST node for its operation tag and must
        return ``(inferred_type, substitution)`` according to Physika's ``type`` system.

        Returns
        -------
        dict[str, Callable]
            Mapping of ``{op_tag: handler}``.

        Examples
        --------
        >>> from physika.elf import ELF
        >>> from physika.utils.infer_expr import infer_expr
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def type_rules(self):
        ...         def check(node, env, s, func_env, class_env, add_error, infer_expr):
        ...             _, cond, body = node
        ...             cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
        ...             if cond_t != ("scalar",):
        ...                 add_error("while condition must be scalar")
        ...             return None, s
        ...         return {"while_loop": check}
        >>> from physika.utils.type_checker_utils import Substitution
        >>> check = WhileLoopFeature().type_rules()["while_loop"]
        >>> node = ("while_loop", ("var", "done"), [])
        >>> errors = []
        >>> t, s = check(node, {"done": ("scalar",)}, Substitution(), {}, {}, errors.append, infer_expr)  # noqa: E501
        >>> t is None and errors == []
        True
        """
        return {}

    def forward_rules(self) -> dict:
        """
        Return a mapping from AST node tags to code generation handlers.

        Each handler receives the AST node and emits a
        Python/PyTorch source string.

        Returns
        -------
        dict[str, Callable]
            Mapping of ``{node_tag: handler}``.

        Examples
        --------
        >>> from physika.elf import ELF
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def forward_rules(self):
        ...         def emit(node):
        ...             from physika.utils.ast_utils import generate_statement, condition_to_expr
        ...             _, cond, body = node
        ...             body_lines = [f"    {generate_statement(s, set())}" for s in body]
        ...             body_code = "\\n".join(body_lines) if body_lines else "    pass"
        ...             return f"while {condition_to_expr(cond)}:\\n{body_code}"  # noqa: E501
        ...         return {"while_loop": emit}
        >>> emit = WhileLoopFeature().forward_rules()["while_loop"]
        >>> node = ("while_loop", ("cond_lt", ("var", "n"), ("num", 10.0)), [])
        >>> emit(node)
        'while n < 10.0:\\n    pass'
        """
        return {}

    def backward_rules(self) -> dict:
        """
        Return a mapping from AST node tags to differentiation handlers.

        Each handler receives the AST node and a ``grad_output``
        expression string representing the upstream gradient, and must
        return a Python/PyTorch source string that computes the
        gradient contribution.

        Returns
        -------
        dict[str, Callable]
            Mapping of ``{node_tag: backward_handler}``.

        Examples
        --------
        >>> from physika.elf import ELF
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def backward_rules(self):
        ...         def grad(node, grad_output, **_):
        ...             # Adjoint method: reverse through the recorded tape,
        ...             # applying the body VJP at each step.
        ...             return (
        ...                 f"_adj = {grad_output}\\n"
        ...                 f"for _state in reversed(_while_tape):\\n"
        ...                 f"    _adj = _body_vjp(_state, _adj)"
        ...             )
        ...         return {"while_loop": grad}
        >>> grad_fn = WhileLoopFeature().backward_rules()["while_loop"]
        >>> node = ("while_loop", ("var", "done"), [])
        >>> print(grad_fn(node, "dL_dy"))
        _adj = dL_dy
        for _state in reversed(_while_tape):
            _adj = _body_vjp(_state, _adj)
        """
        return {}
