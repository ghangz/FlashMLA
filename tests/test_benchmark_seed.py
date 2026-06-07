import ast
import random
from pathlib import Path
from unittest.mock import Mock


def _load_seed_helper():
    source_path = Path(__file__).resolve().parents[1] / "benchmark" / "bench_flash_mla.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "random" for alias in node.names)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "set_benchmark_seed")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": Mock()}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def test_set_benchmark_seed_updates_torch_and_python_random():
    namespace = _load_seed_helper()
    namespace["set_benchmark_seed"](123)

    namespace["torch"].manual_seed.assert_called_once_with(123)
    first = random.random()
    random.seed(123)
    assert first == random.random()


if __name__ == "__main__":
    test_set_benchmark_seed_updates_torch_and_python_random()
