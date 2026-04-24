from typing import Any, Callable, Optional, Tuple
<<<<<<< HEAD
from physika.utils.types import Substitution, Type, new_var
=======
from physika.utils.types import Substitution, Type, new_var, T_NAT, new_dim, TVar, TDim
>>>>>>> 0c07a63 (add if-else infer-stmt hanlders and cond exprs)


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
    ...     add_error=mismatch_errors.append, func_name='f', return_type=T_REAL,
    ... )
    >>> stmt_body_decl(mismatch_stmt, ctx2)
    >>> mismatch_errors
    ["In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"]
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
<<<<<<< HEAD
=======

def stmt_body_if_return(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer and check the return expression of an ``if`` return statement.

    The general form of an body if-return node looks like:
    - ``("body_if_return", cond_expr, ret_expr)`` 

    Correspoinding to a phyiska source code of the form::
    def func(x: ℝ) -> ℝ:
        if cond:
            return expr

    If ``ctx.return_type`` is set, the
    inferred type is unified against it and a type-mismatch error is reported
    on failure.

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("body_if_return", cond_expr, ret_expr)``.
    ctx : StmtContext
        Current inference context.  ``ctx.return_type`` must be set (by
        ``check_function``) for the return-type check to fire.

    Returns
    -------
    None
        Updates ``ctx.s`` with any new unification bindings.
        Calls ``ctx.add_error`` if the inferred return type does not match
        ``ctx.return_type``.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_if_return, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        errors=errors)
    >>> cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
    >>> stmt_body_if_return(('body_if_return', cond, ('var', 'x')), ctx)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import unify, type_to_str

    _, cond, ret_expr = stmt
    ctx.infer_type(cond)
    ret_t = ctx.infer_type(ret_expr)
    if ctx.return_type is not None and ret_t is not None:
        try:
            ctx.s = unify(ctx.return_type, ctx.s.apply(ret_t), ctx.s)
        except TypeError as e:
            ctx.add_error(
                f"if-return type mismatch: "
                f"declared {type_to_str(ctx.return_type)}, "
                f"got {type_to_str(ctx.s.apply(ret_t))}: {e}"
            )


def stmt_body_if_else_return(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer and check both branches of an ``if/else`` return statement.

    Handles ``("body_if_else_return", cond_expr, then_expr, else_expr)`` nodes.

    Physika source code would look like:

        if cond:
            return then_expr
        else:
            return else_expr

    Type inference checks for ``then_expr`` and ``else_expr`` types, which are unified
    against each other.  A mismatch here means the two branches disagree on
    what the function returns. Both errors are independent. Then the unified branch (with the inferred type) is 
    unified with the declared type.

    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("body_if_else_return", cond_expr, then_expr, else_expr)``.
    ctx : StmtContext
        Current inference context.  ``ctx.return_type`` must be set for the
        return-type check to fire.

    Returns
    -------
    None
        Updates ``ctx.s`` with any new unification bindings.
        Calls ``ctx.add_error`` for each failed unification.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_if_else_return, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        errors=errors)
    >>> cond = ("gt", ("var", "x"), ("num", 0.0))
    >>> stmt = ("body_if_else_return", cond, ("var", "x"),
    ...         ("num", 0.0))
    >>> stmt_body_if_else_return(stmt, ctx)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import unify, type_to_str

    _, cond, then_expr, else_expr = stmt
    ctx.infer_type(cond)
    then_t = ctx.infer_type(then_expr)
    else_t = ctx.infer_type(else_expr)

    # Phase 1: branch consistency — then and else must agree
    if then_t is not None and else_t is not None:
        try:
            ctx.s = unify(ctx.s.apply(then_t), ctx.s.apply(else_t), ctx.s)
        except TypeError as e:
            ctx.add_error(
                f"if/else branch type mismatch: "
                f"then={type_to_str(ctx.s.apply(then_t))}, "
                f"else={type_to_str(ctx.s.apply(else_t))}: {e}"
            )

    # Phase 2: unified branch type must match declared return type
    unified_branch = ctx.s.apply(then_t) if then_t is not None else (
        ctx.s.apply(else_t) if else_t is not None else None
    )
    if ctx.return_type is not None and unified_branch is not None:
        try:
            ctx.s = unify(ctx.return_type, unified_branch, ctx.s)
        except TypeError as e:
            ctx.add_error(
                f"if/else return type mismatch: "
                f"declared {type_to_str(ctx.return_type)}, "
                f"got {type_to_str(unified_branch)}: {e}"
            )
>>>>>>> 0c07a63 (add if-else infer-stmt hanlders and cond exprs)
