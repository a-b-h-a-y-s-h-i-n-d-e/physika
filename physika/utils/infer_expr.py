from typing import Any, Callable, Optional, Tuple
from physika.utils.types import Substitution, Type, T_REAL, T_COMPLEX


class ExprContext:
    """
    Data class represeting the context in which an expression is being inferred.

    Passed to every ``expr_*`` handler so each handler receives
    the full typing environment.

    Attributes
    ----------
    env : dict
        Maps variable names to their current ``Type``.
    s : Substitution
        The substitution dictionary accumulated so far. Handler functions thread ``s``
        so each step registers any bindings made by previous steps.
    func_env : dict
        Maps user defined function names to ``(param_types, return_type)``.
    class_env : dict
        Maps class names to their definition dicts (``class_params``,
        ``return_type``, ...).
    add_error : Callable
        Error callback.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, T_REAL
    >>> from physika.utils.types import Substitution
    >>> errors = []
    >>> ctx = ExprContext(env={"x": T_REAL}, s=Substitution(), func_env={}, class_env={}, add_error=errors.append)  # noqa: E501
    >>> ctx.env
    {'x': ℝ}
    >>> ctx.s
    {}
    """

    def __init__(self, env: dict, s: Substitution, func_env: dict,
                 class_env: dict, add_error: Callable) -> None:
        self.env = env
        self.s: Substitution = s
        self.func_env: dict = func_env
        self.class_env: dict = class_env
        self.add_error: Callable = add_error


def expr_num(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    The type of a numeric literal is always ``ℝ``.

    Parameters
    ----------
    node : tuple
        AST node of the form ``("num", value)`` where *value* is an
        ``int`` or ``float``.
    ctx : ExprContext
        Current inference context.

    Returns
    -------
    tuple[Type, Substitution]
        Always ``(T_REAL, ctx.s)``.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_num, T_REAL
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_num(("num", 3.14), ctx)
    >>> t
    ℝ
    """
    return T_REAL, ctx.s


def expr_imaginary(node: Any,
                   ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of the imaginary unit ``i``.

    Inside ``for i : Fin(n)`` body ``i`` is bound as a loop index (``ℝ``). But
    at the top level it is the imaginary unit ``ℂ``.

    Parameters
    ----------
    node : tuple
        AST node ``("imaginary",)``.
    ctx : ExprContext
        Current inference context.  When ``"i"`` is present in ``ctx.env``
        the loop variable shadows the imaginary unit.

    Returns
    -------
    tuple[Type, Substitution]
        ``(T_REAL, ctx.s)`` when ``"i"`` is a live loop variable;
        ``(T_COMPLEX, ctx.s)`` otherwise.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_imaginary, T_REAL, T_COMPLEX
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_imaginary(("imaginary",), ctx)
    >>> t
    ℂ
    >>> ctx_loop = ExprContext({"i": T_REAL}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_imaginary(("imaginary",), ctx_loop)  # loop var shadows ℂ
    >>> t
    ℝ
    """
    if "i" in ctx.env:
        # Loop variable shadows the imaginary unit inside for-expr bodies.
        return ctx.s.apply(ctx.env["i"]), ctx.s
    return T_COMPLEX, ctx.s


def expr_var(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Look up a variable in the current environment.

    Returns ``(None, s)`` when the variable is not yet in scope.

    Parameters
    ----------
    node : tuple
        AST node ``("var", name)`` where *name* is the variable name.
    ctx : ExprContext
        Current inference context.  ``ctx.env`` looks for *name* and
        ``ctx.s`` is applied to the result to expose any resolved
        unification bindings.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(resolved_type, ctx.s)`` when *name* is in scope.
        ``(None, ctx.s)`` otherwise.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_var
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_var(("var", "x"), ctx)
    >>> t
    ℝ[3]
    >>> t, _= expr_var(("var", "y"), ctx)  # not in scope
    >>> t is None
    True
    """
    t = ctx.env.get(node[1])
    # Apply pending substitutions so the caller sees the most-resolved type.
    return (ctx.s.apply(t), ctx.s) if t is not None else (None, ctx.s)
