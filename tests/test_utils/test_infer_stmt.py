from physika.utils.types import (
    TTensor,
    T_REAL,
    T_NAT,
    T_COMPLEX,
    TVar,
    Substitution,
)
from physika.utils.infer_stmts import (
    StmtContext,
    stmt_body_decl,
    stmt_body_assign,
)


class TestStmtContext:
    """
    Tests for ``StmtContext``
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
        func_name = 'f'
        return_type = T_REAL
        ctx = StmtContext(env, s, func_env, class_env, cb, func_name,
                          return_type)
        assert ctx.env is env
        assert ctx.s is s
        assert ctx.func_env is func_env
        assert ctx.class_env is class_env
        assert ctx.add_error is cb

    def test_empty_dicts(self):
        """
        All dict arguments may be empty.
        """
        ctx = StmtContext(env={},
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
        ctx = StmtContext(env={},
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
        ctx = StmtContext(env={},
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
        ctx = StmtContext(env=env,
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        env["y"] = T_NAT
        assert ctx.env["y"] == T_NAT


# helper function to create a StmtContext object
def make_stmt_ctx(env=None,
                  s=None,
                  func_env=None,
                  class_env=None,
                  errors=None,
                  func_name=None,
                  return_type=None):
    """
    Build an StmtContext with sensible defaults for unit tests.
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
    if func_name is None:
        func_name = ''

    return StmtContext(
        env=env,
        s=s,
        func_env=func_env,
        class_env=class_env,
        add_error=errors.append,
        func_name=func_name,
        return_type=return_type,
    )


class TestInferTypeMethod:
    """Test StmtContext.infer_type infers correct types for expressions."""

    def test_numeric(self):
        """Numeric expressions infers to ℝ."""
        ctx = make_stmt_ctx()
        assert ctx.infer_type(("num", 3.14)) == T_REAL

        ctx = make_stmt_ctx()
        assert ctx.infer_type(("num", 0)) == T_REAL

        # lookup from env
        ctx = make_stmt_ctx(env={"x": T_REAL})
        assert ctx.infer_type(("var", "x")) == T_REAL

    def test_tensor_var(self):
        """Tensor variable returns its tensor type."""
        vec = TTensor(((3, "invariant"), ))
        ctx = make_stmt_ctx(env={"v": vec})
        assert ctx.infer_type(("var", "v")) == vec

    def test_add_expression(self):
        """Addition of two scalars infers to ℝ."""
        ctx = make_stmt_ctx(env={"x": T_REAL})
        assert ctx.infer_type(("add", ("var", "x"), ("num", 1.0))) == T_REAL

    def test_substitution_updated(self):
        """self.s is the same object after inference."""
        ctx = make_stmt_ctx()
        s_before = ctx.s
        ctx.infer_type(("num", 1.0))
        assert ctx.s is s_before

    def test_unknown_variable_returns_none(self):
        """Variable not in env returns None without raising."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        t = ctx.infer_type(("var", "unknown"))
        assert t is None


class TestStmtBodyDecl:
    """Test function body declaration statements (stmt_body_decl)."""

    def test_add_expr(self):
        """Inferred ℝ matches declared ℝ"""
        stmt = ('body_decl', 'x', 'ℝ', ('add', ('num', 3), ('var', 'x')))
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_env={'f': (['ℝ'], 'ℝ')},
                            errors=errors)
        stmt_body_decl(stmt, ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_numeric_literal(self):
        """Declaring a variable with a numeric literal"""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'y', 'ℝ', ('num', 42.0)), ctx)
        assert ctx.env['y'] == T_REAL
        assert errors == []

    def test_tensor_declared_and_inferred(self):
        """Inferred array ℝ[3] matches declared ℝ[3]."""
        errors = []
        a_type = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]),
                ('array', [('num', 1.0), ('num', 2.0), ('num', 3.0)]))
        stmt_body_decl(stmt, ctx)
        assert ctx.env['v'] == a_type
        assert errors == []

    def test_no_declared_type(self):
        """No type annotation, but env dict gets the inferred type."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'z', None, ('num', 7.0)), ctx)
        assert ctx.env['z'] == T_REAL
        assert errors == []

    def test_type_mismatch_reports_error(self):
        """Declared ℝ[3] but infers ℝ."""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 2.0))
        stmt_body_decl(stmt, ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with"
            " scalar ℝ"
        ]
        # if mismatch, env gets the inferred type
        assert ctx.env['v'] == T_REAL

    def test_type_mismatch(self):
        """Declared ℝ but infers ℝ[2]"""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        stmt = ('body_decl', 'x', 'ℝ', ('array', [('num', 1.0), ('num', 2.0)]))
        stmt_body_decl(stmt, ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'x' declared ℝ, inferred ℝ[2]: Cannot unify scalar ℝ with"
            " tensor ℝ[2]"
        ]

    def test_env_updated(self):
        """Variable is accessible in env for subsequent expressions."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'a', 'ℝ', ('num', 1.0)), ctx)
        assert ctx.env['a'] == T_REAL
        stmt_body_decl(
            ('body_decl', 'b', 'ℝ', ('add', ('var', 'a'), ('num', 2.0))), ctx)
        assert ctx.env['b'] == T_REAL
        assert errors == []

    def test_index(self):
        """
        Indexing a ℝ[3,4] matrix infers to ℝ[4] — declared ℝ[4] matches.
        """
        errors = []
        mat = TTensor(((3, 'invariant'), (4, 'invariant')))
        ctx = make_stmt_ctx(env={'A': mat}, errors=errors)
        stmt_body_decl(('body_decl', 'r', ('tensor', [(4, 'invariant')]),
                        ('index', 'A', ('num', 0.0))), ctx)
        assert ctx.env['r'] == TTensor(((4, 'invariant'), ))
        assert errors == []

    def test_slice(self):
        """Slicing ℝ[6] with literal bounds infers ℝ[3] type"""
        errors = []
        vec = TTensor(((6, 'invariant'), ))
        sliced = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, errors=errors)
        stmt_body_decl(('body_decl', 's', ('tensor', [(3, 'invariant')]),
                        ('slice', 'v', ('num', 1.0), ('num', 4.0))), ctx)
        assert ctx.env['s'] == sliced
        assert errors == []

    def test_for_expr(self):
        """for i : ℕ(3) infers ℝ[3] type"""
        errors = []
        vec3 = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'v', ('tensor', [(3, 'invariant')]),
                        ('for_expr', 'i', ('num', 3.0), ('num', 1.0))), ctx)
        assert ctx.env['v'] == vec3
        assert errors == []

    def test_index_mismatch(self):
        """Declared ℝ[4] but index of ℝ[4] yields ℝ — mismatch reported."""
        errors = []
        vec = TTensor(((4, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, func_name='f', errors=errors)
        stmt_body_decl(('body_decl', 'x', ('tensor', [(4, 'invariant')]),
                        ('index', 'v', ('num', 0.0))), ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'x' declared ℝ[4], inferred ℝ: Cannot unify tensor ℝ[4] with scalar ℝ"  # noqa: E501
        ]


class TestStmtBodyAssign:
    """Test function body assignment statements (stmt_body_assign)."""

    def test_scalar_assignment(self):
        """Assigning a numeric literal registers ℝ in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'x', ('num', 3.0)), ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_add_expression(self):
        """Assigning a scalar addition registers ℝ in env."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_body_assign(
            ('body_assign', 'y', ('add', ('var', 'x'), ('num', 1.0))), ctx)
        assert ctx.env['y'] == T_REAL
        assert errors == []

    def test_array_assignment(self):
        """Assigning an array literal registers tensor type in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'v', ('array', [('num', 1.0),
                                                         ('num', 2.0),
                                                         ('num', 3.0)])), ctx)
        assert ctx.env['v'] == TTensor(((3, 'invariant'), ))
        assert errors == []

    def test_no_type_error(self):
        """stmt_body_assign dont reports errors."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, func_name='f', errors=errors)
        # Assigning a tensor to x even though x was ℝ
        stmt_body_assign(('body_assign', 'x', ('array', [('num', 1.0),
                                                         ('num', 2.0)])), ctx)
        assert errors == []
        assert ctx.env['x'] == TTensor(((2, 'invariant'), ))

    def test_env_updated(self):
        """Assigned variable is used in following assignments."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'a', ('num', 5.0)), ctx)
        stmt_body_assign(
            ('body_assign', 'b', ('add', ('var', 'a'), ('num', 1.0))), ctx)
        assert ctx.env['a'] == T_REAL
        assert ctx.env['b'] == T_REAL
        assert errors == []

    def test_slice(self):
        """
        Assigning a slice of ℝ[6] with literal bounds registers ℝ[3]
        in env.
        """
        errors = []
        vec = TTensor(((6, 'invariant'), ))
        sliced = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, errors=errors)
        stmt_body_assign(
            ('body_assign', 's', ('slice', 'v', ('num', 1.0), ('num', 4.0))),
            ctx)
        assert ctx.env['s'] == sliced
        assert errors == []

    def test_for_expr_scalar_body(self):
        """Assigning for i : ℕ(3) → num registers ℝ[3] in env."""
        errors = []
        vec3 = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'v', ('for_expr', 'i', ('num', 3.0),
                                               ('num', 1.0))), ctx)
        assert ctx.env['v'] == vec3
        assert errors == []

    def test_fresh_var(self):
        """If type cannot be inferred, a fresh type variable is stored."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'z', ('var', 'unknown')), ctx)
        assert ctx.env['z'] == TVar('α2')
