from physika.elf import ELF



# Helper to test superscript lexer and parser rules
SUPER_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}

def decode_dim(s: str) -> int:
    """
    Helper function to decode a string of unicode superscript
    to an integer.

    Parameter
    ---------
    s : str
        A string of unicode superscript digits.
    """
    return int("".join(SUPER_MAP[c] for c in s))


class SuperindexELF(ELF):
    """
    ELF to add superindex lexer and parser rules
    for TTensor types in Physika.


    Added t_TYPESUPER lexer function and a ``type_spec : TYPESUPER``
    parser rule, so the superscript notation is included at 
    physika.lexer.
    """
    name = "superindex"

    def lexer_rules(self):
        def t_TYPESUPER(t):
            """
            L
            """
            r"ℝ[⁰¹²³⁴⁵⁶⁷⁸⁹]+(ˣ[⁰¹²³⁴⁵⁶⁷⁸⁹]+)*"
            parts = t.value[1:].split("ˣ")
            t.value = ("tensor", [decode_dim(p) for p in parts])
            return t
        return {"token_funcs": [t_TYPESUPER], "tokens": ["TYPESUPER"]}

    def parser_rules(self):
        def p_type_tensor_super(p):
            """type_spec : TYPESUPER"""
            # ℝ² -> ("tensor", [2])  
            # ℝ[2] -> ("tensor", [2])
            _, dims = p[1]
            p[0] = ("tensor", dims)
        return [p_type_tensor_super]


class WhileLoopFeature(ELF):
         name = "while_loop"
         def lexer_rules(self):
             return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"]}
         def parser_rules(self):
             def p_while(p):
                 """statement : WHILE condition COLON NEWLINE INDENT statements DEDENT"""
                 p[0] = ("while_loop", p[2], p[6])
             return [p_while]
         def type_rules(self):
             def check(node, env, s, func_env, class_env, add_error, infer_expr):
                 """
                 Type check the while loop condition is scalar.
                 """
                 _, cond, _ = node
                 # checking only the condition for simplicity
                 cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
                 if cond_t != ("scalar",):
                     add_error("while condition must be scalar")
                 return None, s
             return {"while_loop": check}

         def forward_rules(self):
             def emit(node):
                 """
                 Based on the new WhileELF, we can get the ASTNode tag 
                 and emit python code.
                 """
                 from physika.utils.ast_utils import generate_statement, condition_to_expr
                 _, cond, body = node
                 body_lines = [f"    {generate_statement(s, set())}" for s in body]
                 body_code = "\\n".join(body_lines) if body_lines else "    pass"
                 return f"while {condition_to_expr(cond)}:\\n{body_code}"
             return {"while_loop": emit}
         def backward_rules(self):
             def grad(grad_output):
                """
                We need to test gradients are taken correctly through the while loop.
                For simplicity, return a string of the backward code that will be emitted
                in grad call.
                """

                return (
                     f"_adj = {grad_output}\\n"
                     f"for _state in reversed(_while_tape):\\n"
                     f"    _adj = _body_vjp(_state, _adj)"
                 )
             return {"while_loop": grad}
    
class TestELF:
    def test_ELF_methods(self):
        """Test that ELF methods return expected structures."""
        empty_elf = ELF()
        rules = empty_elf.lexer_rules()
        assert rules["reserved"] == {}
        assert rules["tokens"] == []
        assert empty_elf.parser_rules() == []
        assert empty_elf.type_rules() == {}
        assert empty_elf.forward_rules() == {}
        assert empty_elf.backward_rules() == {}

    def test_while_loop_feature(self):
        """Test that WhileLoopFeature emits expected code."""
        feature_name = "while_loop"
        while_forward_emit = WhileLoopFeature().forward_rules()[feature_name]
        node = ("while_loop", ("cond_lt", ("var", "n"), ("num", 10.0)),
                 [("assign", "n", ("add", ("var", "n"), ("num", 1.0)), 1)])
        assert while_forward_emit(node) == 'while n < 10.0:\\n    n = (n + 1.0)'  # noqa: E501


class TestSuperindexELF:
    def test_lexer_method(self):
        """
        Tests that SuperindexELF defines a t_ lexer function
        and produce the expected token.
        """
        rules = SuperindexELF().lexer_rules()
        names = [fn.__name__ for fn in rules["token_funcs"]]
        assert "t_TYPESUPER" in names
        assert "TYPESUPER" in rules["tokens"]

    def test_token(self):
        """
        Test that t_TYPESUPER decodes properly.
        """
        # case ℝ²
        fn = SuperindexELF().lexer_rules()["token_funcs"][0]
        class TYPE:
            value = "ℝ²"
        assert fn(TYPE()).value == ("tensor", [2])

        # case ℝ²ˣ¹
        fn = SuperindexELF().lexer_rules()["token_funcs"][0]
        class TYPE:
            value = "ℝ²ˣ¹"
        assert fn(TYPE()).value == ("tensor", [2, 1])

    def test_parser_rule(self):
        """
        Tests that the parser rule for superscripted types is defined
        as PLY grammar rule.
        """
        super_index_elf = SuperindexELF()
        assert len(super_index_elf.parser_rules()) == 1
        super_index_string = super_index_elf.parser_rules()[0].__doc__
        assert super_index_string.strip() == "type_spec : TYPESUPER"  # noqa: E501

    def test_parser_rule_produces_tensor_type(self):
        # In PLY, p[0] is the result, and
        # p[1] is the matched token value
        rule_fn = SuperindexELF().parser_rules()[0]
        p = [None, ("tensor", [3, 4])]
        rule_fn(p)
        assert p[0] == ("tensor", [3, 4])

    def test_superscript_matches_bracket_ast(self):
        """
        Tests that the AST produced by parsing a type with superscript matches
        the AST produced by parsing the equivalent type with brackets.
        """
        fn = SuperindexELF().lexer_rules()["token_funcs"][0]
        rule_fn = SuperindexELF().parser_rules()[0]
        class TYPE:
            value = "ℝ²"

        p = [None, fn(TYPE()).value]
        rule_fn(p)
        assert p[0] == ("tensor", [2])


class TestWhileLoopELF:
    feature = WhileLoopFeature()

    def test_lexer(self):
        """
        Test that the lexer rules include the reserved keyword and token name.
        """
        # reserved keyword
        rules = self.feature.lexer_rules()
        assert rules["reserved"]["while"] == "WHILE"

        # token name
        rules = self.feature.lexer_rules()
        assert "WHILE" in rules["tokens"]
        rules = self.feature.lexer_rules()
        assert "WHILE" in rules["tokens"]
    
    def test_parser_rules(self):
        """
        Checks parser rules are defined with correct PLY grammar.
        """
        assert len(self.feature.parser_rules()) == 1
        while_loop_string = self.feature.parser_rules()[0].__doc__
        assert while_loop_string.strip() == "statement : WHILE condition COLON NEWLINE INDENT statements DEDENT"  # noqa: E501

    def test_type_rule(self):
        """
        Test that the type rule for while_loop checks the condition is scalar
        """
        from physika.utils.type_checker_utils import Substitution
        from physika.utils.infer_expr import infer_expr

        assert "while_loop" in self.feature.type_rules()

        check = self.feature.type_rules()["while_loop"]
        
        errors = []
        t, _ = check(
            ("while_loop", ("var", "n"), []),
            {"n": ("scalar",1)}, Substitution(), {}, {},
            errors.append, infer_expr,
        )
        assert t is None
        assert errors == []

        # non-scalar condition produce an error
        errors = []
        check(
            ("while_loop", ("var", "v"), []),
            {"v": ("tensor", [3])}, Substitution(), {}, {},
            errors.append, infer_expr,
        )
        assert len(errors) == 1
        assert errors[0] == "while condition must be scalar"

    def test_forward_rule_key(self):
        """
        Verifies ASTNode tag is correct for WhileLoopFeature and 
        the generated code is as expected.
        """
        assert "while_loop" in self.feature.forward_rules()

        emit = self.feature.forward_rules()["while_loop"]
        # common AST body assignment node
        body = [("assign", "n", ("add", ("var", "n"), ("num", 1.0)), 1)]

        # test code generation with new defined tag
        node = ("while_loop", ("cond_lt", ("var", "n"), ("num", 10.0)), body)
        assert emit(node) == "while n < 10.0:\\n    n = (n + 1.0)"

    def test_backward_rule_key(self):
        """
        Verifies the backward rule is defined for the correct ASTNode tag and
        the emitted code matches the expected backward pass code."""
        assert "while_loop" in self.feature.backward_rules()

        grad = self.feature.backward_rules()["while_loop"]
        result = grad("dL_dn")
   
        assert result == "_adj = dL_dn\\nfor _state in reversed(_while_tape):\\n    _adj = _body_vjp(_state, _adj)"  # noqa: E501