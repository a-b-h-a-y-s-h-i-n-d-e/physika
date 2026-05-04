Easy Language Feature (ELF)
===========================

Every Physika program is processed through a pipeline of parser and lexer rules that produce an Abstract Syntax Tree (AST).
Physika's type system then verifies each AST node for type correctness by comparing inferred types with declared types.
If no type errors are found, the AST is passed to the code generator, which emits PyTorch/Python code as strings for both forward and backward passes.
ELFs allow new rules to be easily added at each stage of this pipeline, parsing, type checking, and code generation, without modifying the core Physika codebase, which facilitate implementation and maintenance.

Each ``ELF`` subclass declares a unique ``name`` and overrides four methods that control its behavior in Physika:

* **Parser rules**: PLY grammar functions that introduce new syntax rules.
* **Lexer rules**: New reserved keywords and token names for the lexer.
* **Type rules**: Physika's type checker handlers that receive AST nodes and update the type environment following Hindley-Milner type inference.
* **Forward rules**: Code generation handlers that emit PyTorch code as strings from AST nodes.
* **Backward rules**: Differentiation handlers for custom gradient computation.

Feature Registry
----------------

Once an ELF is defined, its rules must be registered so that each Physika module can use them. This is handled by ``FeatureRegistry``, which stores incoming rules and dispatches them to the appropriate pipeline stage.
``FeatureRegistry`` class have seven methods:

* ``register``: Accepts an ELF instance and stores its rules in dispatch tables keyed by ``ELF.name``.
* ``add_lexer_rules``: Adds new PLY tokens and reserved keywords to ``physika.lexer``. If any function-based token rules are added, the lexer is rebuilt and the updated instance is swapped into the ``IndentLexer`` wrapper class at ``physika.lexer``.
* ``add_parser_rules``: Injects PLY grammar functions into ``physika.parser`` so that ``yacc.yacc()`` register them.
* ``has_type_rule``: Returns ``True`` if a type inference handler is registered for a given AST node tag.
* ``dispatch_type``: Calls the registered type inference handler for an AST node tag and returns its inferred type.
* ``dispatch_forward``: Calls the registered code generation handler for an AST node tag and returns the emitted Python source string.
* ``dispatch_backward``: Calls the registered differentiation handler for an AST node tag and returns the emitted Pytorch gradient code.

Example: While Loop Feature
----------------------------

The following example implements a ``while_loop`` statement as a complete ELF,
exercising all five rule types: lexer, parser, type, forward, and backward.

.. code-block:: python

    from physika.elf import ELF
    from physika.utils.ast_utils import generate_statement, condition_to_expr

    class WhileLoopFeature(ELF):
        name = "while_loop"

        def lexer_rules(self):
            # Adds "while" as a reserved keyword mapped to the WHILE token.
            return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"]}

        def parser_rules(self):
            def p_while(p):
                """statement : WHILE condition COLON NEWLINE INDENT statements DEDENT"""
                p[0] = ("while_loop", p[2], p[6])
            return [p_while]

        def type_rules(self):
            def check(node, env, s, func_env, class_env, add_error, infer_expr):
                _, cond, _ = node
                cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
                if cond_t != ("scalar",):
                    add_error("while condition must be scalar")
                return None, s
            return {"while_loop": check}

        def forward_rules(self):
            def emit(node):
                _, cond, body = node
                body_lines = [f"    {generate_statement(s, set())}" for s in body]
                body_code = "\n".join(body_lines) if body_lines else "    pass"
                return f"while {condition_to_expr(cond)}:\n{body_code}"
            return {"while_loop": emit}

        def backward_rules(self):
            def grad(grad_output):
                # Adjoint method: reverse through the recorded tape,
                # applying the body VJP at each step.
                return (
                    f"_adj = {grad_output}\n"
                    f"for _state in reversed(_while_tape):\n"
                    f"    _adj = _body_vjp(_state, _adj)"
                )
            return {"while_loop": grad}

Once the ELF is defined, register it with ``FeatureRegistry`` and use the dispatch
methods to type check and generate code for ``while_loop`` nodes (Each registry will occur in the appropiate file path):

.. code-block:: python

    # At __init__.py of ELFs dir
    from physika.elf import FeatureRegistry

    reg = FeatureRegistry()
    reg.register(WhileLoopFeature())

.. code-block:: python

    # At physika/utils/types.py
    # Check that type and forward rules were registered
    reg.has_type_rule("while_loop")   # True

.. code-block:: python

    # At physika/utils/ast_utils.py
    # Forward dispatch: emit Python code for a while_loop AST node
    # Parser rules will add a node like this to AST:
    # node = (
    #    "while_loop",
    #    ("cond_lt", ("var", "n"), ("num", 10.0)),
    #    [("assign", "n", ("add", ("var", "n"), ("num", 1.0)), 1)],
    #)
    reg.dispatch_forward("while_loop", node)
    # 'while n < 10.0:\n    n = (n + 1.0)'

    # Backward dispatch: emit adjoint gradient code
    reg.dispatch_backward("while_loop", "dL_dn")
    # '_adj = dL_dn\nfor _state in reversed(_while_tape):\n    _adj = _body_vjp(_state, _adj)'
