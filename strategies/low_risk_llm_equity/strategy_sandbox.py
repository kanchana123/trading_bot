import ast
from typing import Dict, Tuple

ALLOWED_IMPORT_ROOTS = {
    "pandas",
    "numpy",
    "pandas_ta",
    "math",
    "datetime",
    "typing",
    "statistics",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
    "exit",
    "quit",
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Disallowed import: {name}")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": _safe_import,
}


class _StrategySafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed import: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            raise ValueError(f"Disallowed import from: {node.module}")

    def visit_Call(self, node: ast.Call):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FORBIDDEN_CALLS:
            raise ValueError(f"Disallowed call: {name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError(f"Disallowed attribute access: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "requests"}:
            raise ValueError(f"Disallowed name: {node.id}")
        self.generic_visit(node)


def compile_strategy_code(strategy_code: str, extra_globals: Dict) -> Tuple[Dict, Dict]:
    """Parse, safety-check, and exec LLM-generated strategy code in a restricted scope."""
    tree = ast.parse(strategy_code)
    _StrategySafetyVisitor().visit(tree)
    compiled = compile(tree, filename="<llm_strategy>", mode="exec")
    execution_globals = {"__builtins__": dict(SAFE_BUILTINS)}
    execution_globals.update(extra_globals)
    strategy_scope: Dict = {}
    exec(compiled, execution_globals, strategy_scope)
    required = ("calculate_indicators", "should_buy", "should_sell")
    missing = [name for name in required if not callable(strategy_scope.get(name))]
    if missing:
        raise ValueError(f"Generated strategy is missing functions: {', '.join(missing)}")
    return execution_globals, strategy_scope
