from typing import Any, Callable, Optional, Tuple
from physika.utils.types import Substitution, Type, new_var


class StmtContext:
    """
    Data class that contains the context of a Physika statement that is being
    inferred.

    ``StmtContext`` is passed to every ``stmt_*`` inference statement handler.

    Attributes
    ----------
    env : dict
        Maps variable names to their current types
    s : Substitution
        Dicitionary accumulated with valid substitutions. Each statement
        handler aims to resolve any unknown types using previous bindings
        or adding new ones.
    func_env : dict
        Maps user defined function names to ``(param_types, return_type)``.
    class_env : dict
        Maps class names to their definition dicts (``class_params``,
        ``return_type``, ...).
    add_error : Callable
        Error callback.
    func_name : str
        User defined function name for checking check body statements. This
        field is used when calling ``check_function`` from main type
        checking algorithm.
    return_type : Optional[Type]
        Used especifically in ``body_if_return`` and ``body_if_else_return``
        to unify the return expression type against it. Propagated into nested
        function body scopes so that return statements inside can still
        be checked.

    Examples
    --------
    >>> from physika.utils.infer_stmts import StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={}, s=Substitution(), func_env={}, class_env={},
    ...     add_error=errors.append, func_name="f", return_type=T_REAL,
    ... )
    >>> t = ctx.infer_type(("num", 1.0))
    >>> t
    ℝ
    >>> errors
    []
    """

    def __init__(self,
                 env: dict,
                 s: Substitution,
                 func_env: dict,
                 class_env: dict,
                 add_error: Callable,
                 func_name: str = "?",
                 return_type: Optional[Type] = None) -> None:
        self.env: dict = env
        self.s: Substitution = s
        self.func_env: dict = func_env
        self.class_env: dict = class_env
        self.add_error: Callable = add_error
        self.func_name: str = func_name
        self.return_type: Optional[Type] = return_type

    def infer_type(self, expr: Any) -> Optional[Type]:
        """Infer the type of a Physika expression.

        Calls ``infer_expr`` using the current context environments and
        updates ``self.s`` in place with new bindings if any.

        Parameters
        ----------
        expr : Any
            A Physika AST expression node  such as
            ``("add", left, right)``, ``("call", name, args)``,
            ``("index", arr, idx)``, ``("for_expr", var, size, body)``,
            or a numeric literal (``int`` / ``float``).

        Returns
        -------
        Optional[Type]
            The inferred ``Type`` (e.g. ``T_REAL``),
            or ``None`` if the expression is not resolved (unknown variable)


        Examples
        --------
        >>> from physika.utils.infer_stmts import StmtContext
        >>> from physika.utils.types import Substitution, T_REAL, TTensor
        >>> errors = []
        >>> ctx = StmtContext(
        ...     env={"v": TTensor(((3, "invariant"),))}, s=Substitution(), func_env={}, class_env={},  # noqa: E501
        ...     add_error=errors.append, func_name="f", return_type=T_REAL,
        ... )
        >>> ctx.infer_type(("num", 2.0))
        ℝ
        >>> ctx.infer_type(("var", "v"))
        ℝ[3]
        >>> errors
        []
        """
        from physika.utils.infer_expr import infer_expr

        t, self.s = infer_expr(expr, self.env, self.s, self.func_env,
                               self.class_env, self.add_error)
        return t


def stmt_body_decl(stmt: Tuple, ctx: StmtContext) -> None:
    """
    Infer type of declaration statements inside functions.

    The inferred type is obtained by calling ``ctx.infer_type`` on the expression ``expr`` found
    in ``body_decl`` statement. Then, the inferred type is unified with declared type including the updated bindings
    at ``s: Substitution`` . If there is a mismatch, an error is reported.

    Parameters
    ----------
    stmt: tuple
        AST node of the form ``("body_decl", 'var', 'var_type', expr)`` where *expr* can be any
        for of supported expressions at ``infer_expr``.
    ctx: StmtContext
        Current inference context

    Returns
    -------
    None
        If there is any type mismatch between inferred and declared, report an
        error and add to ``env`` the inferred type if not None. Else, add a
        new unknown var type. If there is no mismatch, add declared type at ``env``.
        If declared type is None, add inferred type if known, else add unknown var type.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_decl, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> stmt = ('body_decl', 'x', 'ℝ', ('add', ('num', 3), ('var', 'x')))
    >>> s = Substitution()
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={'x': T_REAL},
    ...     s=s,
    ...     func_env={'f': (['ℝ'], 'ℝ')},
    ...     class_env={},
    ...     add_error=errors.append,
    ...     func_name='f',
    ...     return_type=T_REAL)
    >>> stmt_body_decl(stmt, ctx)
    >>> print(errors)
    []
    >>> mismatch_stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 2.0))  # noqa: E501
    >>> mismatch_errors = []
    >>> ctx2 = StmtContext(
    ...     env={}, s=Substitution(), func_env={}, class_env={},
    ...     add_error=mismatch_errors.append, func_name='f', return_type=T_REAL,  # noqa: E501
    ... )
    >>> stmt_body_decl(mismatch_stmt, ctx2)
    >>> mismatch_errors
    ["In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"]  # noqa: E501
    """
    from physika.utils.type_checker_utils import from_typespec, unify, type_to_str  # noqa: E501

    # example stmt node: ('body_decl', var, var_type, expr)
    _, var_name, var_type_spec, expr = stmt
    inferred = ctx.infer_type(expr)
    declared = from_typespec(var_type_spec)
    mismatch = False
    if declared is not None and inferred is not None:
        try:
            ctx.s = unify(declared, inferred, ctx.s)
        except TypeError as e:
            mismatch = True
            ctx.add_error(
                f"In '{ctx.func_name}': '{var_name}' declared {type_to_str(declared)}, "  # noqa: E501
                f"inferred {type_to_str(ctx.s.apply(inferred))}: {e}")

    # Update env dictionary
    if mismatch:
        if inferred is not None:
            ctx.env[var_name] = inferred
        else:
            ctx.env[var_name] = new_var()
    else:
        # No mismatch and declared exists
        if declared is not None:
            ctx.env[var_name] = declared
        else:
            # Add inferred value at env, or create a new variable
            # if none exists
            ctx.env[var_name] = inferred or new_var()


def stmt_body_assign(stmt: Any, ctx: StmtContext) -> None:
    """
    Inferred type of assingment statements inside functions.

    ``stmt_body_assign`` is similar to ``stmt_body_decl``, but in
    ``body_assign`` statements ``type_var`` is not declared. However,
    it is inferred and used for following unification until function
    return.

    If inference returns ``None`` (unknown expression), a new ``TVar``
    type variable is stored instead.

    Parameters
    ----------
    stmt: tuple
        AST node of the form ``("body_assign", var_name, expr)``.
    ctx: StmtContext
        Current inference context.

    Returns
    -------
    None
        Updates ``ctx.env[var_name]`` in place.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_assign, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_env={},
    ...                   class_env={}, add_error=errors.append)
    >>> stmt_body_assign(('body_assign', 'y', ('add', ('var', 'x'), ('num', 1.0))), ctx)  # noqa: E501
    >>> ctx.env['y']
    ℝ
    >>> errors
    []
    >>> ctx2 = StmtContext(env={}, s=Substitution(), func_env={}, class_env={},
    ...                    add_error=errors.append, return_type=T_REAL)
    >>> stmt_body_assign(('body_assign', 'v', ('array', [('num', 1.0), ('num', 2.0)])), ctx2)  # noqa: E501
    >>> ctx2.env['v']
    ℝ[2]
    >>> errors
    []
    """
    # assing statement example node: ("body_assign", var_name, expr)
    _, var_name, expr = stmt
    inferred = ctx.infer_type(expr)
    ctx.env[var_name] = inferred if inferred is not None else new_var()
