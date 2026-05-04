import logging
import types
from typing import Any, Callable, Optional
import ply.lex as ply_lex  # type: ignore


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


class FeatureRegistry:
    """
    Collects registered ELF features, adds their lexer and parser rules at
    import time, and dispatches type inference and code generation handlers
    at runtime.

    Examples
    --------
    >>> from physika.elf import ELF, FeatureRegistry
    >>> class WhileLoopFeature(ELF):
    ...     name = "while_loop"
    ...     def forward_rules(self):
    ...         def emit(node, **ctx):
    ...             _, cond, _ = node
    ...             return f"while {cond}:\\n    pass"
    ...         return {"while_loop": emit}
    ...     def backward_rules(self):
    ...         def grad(node, g, **kw): return f"_adj = {g}"
    ...         return {"while_loop": grad}
    >>> reg = FeatureRegistry()
    >>> reg.register(WhileLoopFeature())
    >>> reg.dispatch_forward("while_loop", ("while_loop", "x > 0", []))
    'while x > 0:\\n    pass'
    >>> reg.dispatch_backward("while_loop", ("while_loop", "x > 0", []), "dL")
    '_adj = dL'
    """

    def __init__(self) -> None:
        """
        Initialize empty dispatch tables.

        Attributes
        ----------
        features : list[ELF]
            Ordered list of registered ELF feature instances.
        type_dispatch : dict[str, Callable]
            Maps AST node tags to type inference handlers.
        forward_dispatch : dict[str, Callable]
            Maps AST node tags to forward code generation handlers.
        backward_dispatch : dict[str, Callable]
            Maps AST node tags to backward differentiation handlers.

        Examples
        --------
        >>> from physika.elf import FeatureRegistry
        >>> reg = FeatureRegistry()
        >>> reg.features
        []
        >>> reg.type_dispatch, reg.forward_dispatch, reg.backward_dispatch
        ({}, {}, {})
        """
        self.features: list[ELF] = []
        self.type_dispatch: dict[str, Callable] = {}
        self.forward_dispatch: dict[str, Callable] = {}
        self.backward_dispatch: dict[str, Callable] = {}

    def register(self, feature: ELF) -> None:
        """
        Register a new ELF subclass instance and update rule dispatch tables.

        Parameters
        ----------
        feature : ELF
            ELF subclass to be registerd.

        Returns
        -------
        None
            ``.register()`` updates the registry dictionaries in place.

        Examples
        --------
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def forward_rules(self):
        ...         def emit(node, **ctx):
        ...             _, cond, _ = node
        ...             return f"while {cond}:\\n    pass"
        ...         return {"while_loop": emit}
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> len(reg.features)
        1
        """
        self.features.append(feature)
        self.type_dispatch.update(feature.type_rules())
        self.forward_dispatch.update(feature.forward_rules())
        self.backward_dispatch.update(feature.backward_rules())

    def add_lexer_rules(self, module: types.ModuleType) -> None:
        """
        Adds lexer rules to ``lexer.py`` module from every registered feature.

        For each feature, we have three objects to be merged into lexer module,
        which are: **reserved**, **tokens**, and **token_funcs**.

        ``reserved`` is a dict in ``lexer.py`` that maps keywords
        (e.g. ``"while"``) to their token type (``"WHILE"``). ``tokens``
        are new token name strings (e.g. ``["WHILE"]``), appended to
        ``module.tokens``, avoiding duplicates. ``token_funcs`` are ``t_``
        functions added to ``lexer.py`` so PLY looks for `t_*`` patterns.

        After registering the new lexer additions, ``module.tokens`` is updated.
        Then, if a new function token rule is added, the lexer must be rebuilt
        by running ``ply_lex.lex()`` on the updated module.

        Parameters
        ----------
        module : types.ModuleType
            ``physika.lexer`` module object.

        Examples
        --------
        >>> import types
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def lexer_rules(self):
        ...         return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"], "token_funcs": []}  # noqa: E501
        >>> # mod is actually physika.lexer when running Physika programs
        >>> mod = types.SimpleNamespace(tokens=("ID",), reserved={})
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> reg.add_lexer_rules(mod)
        >>> "WHILE" in mod.tokens
        True
        >>> mod.reserved
        {'while': 'WHILE'}
        """
        tokens_list = list(module.tokens)
        has_new_funcs = False

        for feature in self.features:
            rules = feature.lexer_rules()

            # Includes new reserved tokens to lexer.reserved list.
            module.reserved.update(rules.get("reserved", {}))

            # Append new token names
            for tok in rules.get("tokens", []):
                if tok not in tokens_list:
                    # avoids duplicates.
                    tokens_list.append(tok)

            # adds function token rules to physika.lexer
            # module so PLY can find look for t_* naming convention.
            for fn in rules.get("token_funcs", []):
                setattr(module, fn.__name__, fn)
                has_new_funcs = True  # need a full re-lex below

        # Updates module.tokens
        module.tokens = tuple(tokens_list) # type: ignore

        if has_new_funcs:
            # Rebuild lexer
            err_log = logging.getLogger("elf.lex")
            err_log.propagate = False
            new_raw = ply_lex.lex(module=module, errorlog=err_log)
            # Updates PLY lexer object inside Physika's lexer.py IndentLexer
            # class.
            if hasattr(module, "lexer") and hasattr(module.lexer, "lexer"):
                module.lexer.lexer = new_raw
        else:
            # No new function rules
            # lextokens and lextokens_all are PLY attributes
            if hasattr(module, "lexer") and hasattr(module.lexer, "lexer"):
                inner = module.lexer.lexer
                inner.lextokens = set(module.tokens)
                inner.lextokens_all = set(module.tokens)

    def add_parser_rules(self, module: types.ModuleType) -> None:
        """
        Inject each feature's PLY grammar functions into ``parser.py``.

        Each function returned by ``feature.parser_rules()`` is set as an
        attribute on ``physika.parser``. PLY finds grammar
        rules by looking for ``p_`` attributes at ``yacc.yacc()``.

        Parameters
        ----------
        module : types.ModuleType
            ``physika.parser`` module object.

        Examples
        --------
        >>> import types as types
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def parser_rules(self):
        ...         def p_while(p):
        ...             \"\"\"statement : WHILE expr COLON NEWLINE INDENT stmts DEDENT\"\"\"  # noqa: E501
        ...             p[0] = ("while_loop", p[2], p[6])
        ...         return [p_while]
        >>> # mod is actually physika.lexer when running Physika programs
        >>> mod = types.SimpleNamespace()
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> reg.add_parser_rules(mod)
        >>> hasattr(mod, "p_while")
        True
        >>> mod.p_while.__doc__
        'statement : WHILE expr COLON NEWLINE INDENT stmts DEDENT'
        """
        for feature in self.features:
            for rule_fn in feature.parser_rules():
                setattr(module, rule_fn.__name__, rule_fn)

    def has_type_rule(self, op: str) -> bool:
        """
        Return ``True`` if a type inference handler is registered for
        the new ELF.

        Parameters
        ----------
        op : str
            The node tag to search in AST with the ``.name`` of the subclass
            ELF.

        Examples
        --------
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def type_rules(self):
        ...         return {"while_loop": lambda node, *a, **kw: (None, None)}
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> reg.has_type_rule("while_loop")
        True
        >>> reg.has_type_rule("for_loop")
        False
        """
        return op in self.type_dispatch

    def dispatch_type(self, op: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call the ELF type-inference handler for ``op``.

        The handler is expected to return a tuple of (``inferred_type``, ``substitution``).  # noqa: E501
        If no handler is registered for ``op``, then returns ``None``.

        Parameters
        ----------
        op : str
            The node tag to search in AST with the ``.name`` of the subclass ELF.
        *args, **kwargs
            Arguments to pass to the type inference handler.

        Examples
        --------
        >>> from physika.elf import ELF, FeatureRegistry
        >>> from physika.utils.type_checker_utils import Substitution
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def type_rules(self):
        ...         def check(node, env, s, func_env, class_env, add_error, infer_expr):  # noqa: E501
        ...             return None, s
        ...         return {"while_loop": check}
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> s = Substitution()
        >>> t, _ = reg.dispatch_type("while_loop", ("while_loop", ("var", "n"), []), {}, s, {}, {}, [].append, None)  # noqa: E501
        >>> t is None
        True
        """
        fn = self.type_dispatch.get(op)
        if fn is not None:
            return fn(*args, **kwargs)
        else:
            None

    def dispatch_forward(self, op: str, node: tuple,
                         **ctx: Any) -> Optional[str]:  # noqa: E501
        """
        Call the forward code generation handler for ``op``.

        The handler is expected to return a string of Python code.

        Examples
        --------
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def forward_rules(self):
        ...         def emit(node, **ctx):
        ...             _, cond, _ = node
        ...             return f"while {cond}:\\n    pass"
        ...         return {"while_loop": emit}
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> reg.dispatch_forward("while_loop", ("while_loop", "x > 0", []))
        'while x > 0:\\n    pass'
        """
        fn = self.forward_dispatch.get(op)
        if fn is not None:
            return fn(node, **ctx)
        else:
            return None

    def dispatch_backward(self, op: str, *args: Any,
                          **kwargs: Any) -> Optional[str]:  # noqa: E501
        """
        Call the backward differentiation handler for ``op``, or return ``None``.

        The handler is expected to return a string of PyTorch code that
        computes the gradient for the given node.

        Examples
        --------
        >>> from physika.elf import ELF, FeatureRegistry
        >>> class WhileLoopFeature(ELF):
        ...     name = "while_loop"
        ...     def backward_rules(self):
        ...         def grad(node, g, **kw): return f"_adj = {g}"
        ...         return {"while_loop": grad}
        >>> reg = FeatureRegistry()
        >>> reg.register(WhileLoopFeature())
        >>> reg.dispatch_backward("while_loop", ("while_loop", "x > 0", []), "dL")  # noqa: E501
        '_adj = dL'
        >>> reg.dispatch_backward("unknown_op", ()) is None
        True
        """
        fn = self.backward_dispatch.get(op)
        if fn is not None:
            return fn(*args, **kwargs)
        else:
            return None


REGISTRY = FeatureRegistry()
