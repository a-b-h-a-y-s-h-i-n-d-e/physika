from physika.utils.types import (
    TVar,
    TDim,
    TTensor,
    TFunc,
    T_REAL,
    T_NAT,
    T_COMPLEX,
    Substitution,
)
from physika.utils.infer_expr import (
    ExprContext,
    expr_num,
    expr_imaginary,
    expr_var,
)


class TestExprContext:
    """
    Tests for ``ExprContext``
    """

    def test_fields(self):
        """
        Checks that constructor arguments are stored as attributes.
        """
        env = {"x": T_REAL}
        s = Substitution({"α0": T_NAT})
        func_env = {"f": ([T_REAL], T_REAL)}
        class_env = {"Net": {"class_params": []}}
        errors = []
        cb = errors.append
        ctx = ExprContext(env, s, func_env, class_env, cb)
        assert ctx.env is env
        assert ctx.s is s
        assert ctx.func_env is func_env
        assert ctx.class_env is class_env
        assert ctx.add_error is cb

    def test_empty_dicts(self):
        """
        All dict arguments may be empty.
        """
        ctx = ExprContext(env={},
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        assert ctx.env == {}
        assert ctx.func_env == {}
        assert ctx.class_env == {}

    def test_add_error(self):
        """
        ``add_error`` stores error messages.
        """
        errors = []
        ctx = ExprContext(env={},
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=errors.append)

        ctx.add_error("error #1")
        assert errors == ["error #1"]

        # Multiple errors accumulate in order.
        ctx.add_error("error #2")
        assert errors == ["error #1", "error #2"]

    def test_s_substitution(self):
        """
        ``s`` is stored as the exact Substitution passed in.
        """
        s = Substitution({"α1": T_COMPLEX})
        ctx = ExprContext(env={},
                          s=s,
                          func_env={},
                          class_env={},
                          add_error=[].append)
        assert isinstance(ctx.s, Substitution)
        assert ctx.s["α1"] == T_COMPLEX

    def test_env_mutation(self):
        """
        Mutating the env dict after construction changes ctx.env
        """
        env = {}
        ctx = ExprContext(env=env,
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        env["y"] = T_NAT
        assert ctx.env["y"] == T_NAT


def make_ctx(env=None, s=None, func_env=None, class_env=None, errors=None):
    """
    Build an ExprContext with sensible defaults for unit tests.
    """
    if env is None:
        env = {}
    if s is None:
        s = Substitution()
    if func_env is None:
        func_env = {}
    if class_env is None:
        class_env = {}
    if errors is None:
        errors = []

    return ExprContext(
        env=env,
        s=s,
        func_env=func_env,
        class_env=class_env,
        add_error=errors.append,
    )


class TestExprNum:
    """
    Tests for ``expr_num``.

    Numeric literal inference.
    """

    def test_float(self):
        """Any float literal must infer to ℝ."""
        ctx = make_ctx()
        t, s = expr_num(("num", 3.14), ctx)
        assert t == T_REAL

    def test_int(self):
        """Integer literal refer to ℝ."""
        ctx = make_ctx()
        t, s = expr_num(("num", 42), ctx)
        assert t == T_REAL

        # zero is ℝ
        ctx = make_ctx()
        t, _ = expr_num(("num", 0), ctx)
        assert t == T_REAL

    def test_substitution_unchanged(self):
        """
        The substitution dict does not contain new bindings.
        """
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_num(("num", 1.0), ctx)
        assert s_out == existing

    def test_env_context(self):
        """expr_num infer type with a non-empty environment."""
        ctx = make_ctx(env={"x": T_REAL, "y": TTensor(((3, "invariant"), ))})
        t, _ = expr_num(("num", 7.0), ctx)
        assert t == T_REAL


class TestExprImaginary:
    """Tests for ``expr_imaginary``."""

    def test_ret_complex(self):
        """
        ``i`` is the imaginary unit ℂ.
        """
        ctx = make_ctx()
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_COMPLEX

    def test_loop_var_over_imaginary(self):
        """
        When ``i`` is a live loop variable it shadows ℂ and resolves to ℝ.
        """
        ctx = make_ctx(env={"i": T_REAL})
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_REAL

        # loop variables named ``j``, ``k``, etc,  must not shadow ``i``.
        ctx = make_ctx(env={"j": T_REAL, "k": T_REAL})
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_COMPLEX

    def test_loop_var_substitution_applied(self):
        """
        When ``i`` maps to a TVar that is bound, apply returns the binding.
        """
        alpha = TVar("α0")
        s = Substitution({"α0": T_REAL})
        ctx = make_ctx(env={"i": alpha}, s=s)
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_REAL

    def test_substitution(self):
        """No new bindings are introduced."""
        existing = Substitution({"α1": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_imaginary(("imaginary", ), ctx)
        assert s_out == existing


class TestExprVar:
    """
    Tests for ``expr_var``.

    Variable lookup in the current environment.
    """

    def test_bounded_variable(self):
        """A variable bound to ℝ resolves to ℝ."""
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_var(("var", "x"), ctx)
        assert t == T_REAL

        # a variable bound to ℕ resolves to ℕ.
        ctx = make_ctx(env={"n": T_NAT})
        t, _ = expr_var(("var", "n"), ctx)
        assert t == T_NAT

        # a variable bound to ℝ[3] resolves to ℝ[3].
        vec_t = TTensor(((3, "invariant"), ))
        ctx = make_ctx(env={"v": vec_t})
        t, _ = expr_var(("var", "v"), ctx)
        assert t == vec_t

        # a variable bound to ℝ[2,4] resolves to ℝ[2,4].
        mat_t = TTensor(((2, "invariant"), (4, "invariant")))
        ctx = make_ctx(env={"M": mat_t})
        t, _ = expr_var(("var", "M"), ctx)
        assert t == mat_t

    def test_unbound_variable_returns_none(self):
        """Looking up a name not in scope returns None without error."""
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_var(("var", "y"), ctx)
        assert t is None

    def test_empty_env_returns_none(self):
        """Empty environment always returns None for any name."""
        ctx = make_ctx()
        t, _ = expr_var(("var", "anything"), ctx)
        assert t is None

    def test_substitution_applied_to_tvar(self):
        """
        Checks that if the variable maps to a TVar,
        the substitution is applied.
        """
        alpha = TVar("α0")
        s = Substitution({"α0": T_REAL})
        ctx = make_ctx(env={"x": alpha}, s=s)
        t, _ = expr_var(("var", "x"), ctx)
        assert t == T_REAL

        # symbolic dims in tensor types are resolved through s
        tensor_t = TTensor(((TDim("δ0"), "invariant"), ))
        s = Substitution({"δ0": 5})
        ctx = make_ctx(env={"v": tensor_t}, s=s)
        t, _ = expr_var(("var", "v"), ctx)
        expected = TTensor(((5, "invariant"), ))
        assert t == expected

    def test_bound_substitution(self):
        """
        The substitution is returned unchanged when the variable is found.
        """
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(env={"x": T_REAL}, s=existing)
        _, s_out = expr_var(("var", "x"), ctx)
        assert s_out == existing

        # substitution is returned unchanged even when lookup returns None.
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_var(("var", "missing"), ctx)
        assert s_out == existing

    def test_no_errors_emitted_for_unbound(self):
        """Missing variable does NOT trigger an error — callers handle None."""
        errors = []
        ctx = make_ctx(errors=errors)
        expr_var(("var", "None"), ctx)
        assert errors == []

    def test_func_type_variable(self):
        """A variable holding a function type resolves correctly."""
        f_t = TFunc((T_REAL, ), T_REAL)
        ctx = make_ctx(env={"f": f_t})
        t, _ = expr_var(("var", "f"), ctx)
        assert t == f_t
