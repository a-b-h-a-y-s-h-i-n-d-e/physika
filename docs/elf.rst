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
