import sys

from lexer import lexer
from parser import parser, symbol_table
from type_checker import build_unified_ast, type_check
from codegen import from_ast_to_torch
from utils import print_type_check_results

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        source = f.read()

    # Parse from tokens to AST
    local_program_ast = parser.parse(source, lexer=lexer)
    # Build unified AST
    unified_ast = build_unified_ast(local_program_ast, symbol_table)
    # Type checking
    type_errors = type_check(unified_ast)
    if type_errors:
        print_type_check_results(type_errors)

    # Generate PyTorch code and execute it
    torch_code = from_ast_to_torch(unified_ast, print_code=True)
    exec(torch_code)
