from pathlib import Path

import pytest

from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.codegen import from_ast_to_torch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def exec_phyk(stem: str) -> dict:
    """
    Helper function to execute a .phyk file and return the resulting namespace
    ``ns`` dict.
    """
    source = (EXAMPLES_DIR / f"{stem}.phyk").read_text()
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns


@pytest.fixture(scope="module")
def numeric_ns():
    """
    Execute example_arrays.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_numeric_types")


class TestIntegerType:

    def test_integer_value(self, numeric_ns):
        assert float(numeric_ns["a"]) == 10.0

    def test_integer_addition(self, numeric_ns):
        assert float(numeric_ns["z_add"]) == 13.0


class TestRealType:

    def test_real_value(self, numeric_ns):
        assert float(numeric_ns["x"]) == 3.14

    def test_real_multiplication(self, numeric_ns):
        assert float(numeric_ns["r_mul"]) == 3.14 * 2.0


class TestMixedTypes:

    def test_mixed_values(self, numeric_ns):
        assert float(numeric_ns["z_number"]) == 1.0
        assert float(numeric_ns["r_number"]) == 2.0

    def test_mixed_multiplication(self, numeric_ns):
        assert float(numeric_ns["result"]) == 2.0


class TestNegativeValues:

    def test_negative_integer(self, numeric_ns):
        assert float(numeric_ns["neg_int"]) == -7.0

    def test_negative_real(self, numeric_ns):
        assert float(numeric_ns["neg_float"]) == -3.14
