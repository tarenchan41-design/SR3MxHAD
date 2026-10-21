import ast
from pathlib import Path
from had.model import VARIANTS
from experiments.protocol import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def imported_packages(package):
    found = set()
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if isinstance(node, ast.Import):
                found |= {(path.name, a.name.split(".")[0]) for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add((path.name, node.module.split(".")[0]))
    return found


def test_benchmark_and_method_do_not_depend_on_experiments():
    for package, forbidden in (("sr3m", {"had", "experiments", "vln_benchmark", "source"}),
                               ("had", {"sr3m", "experiments", "vln_benchmark", "source"})):
        bad = {item for item in imported_packages(package) if item[1] in forbidden}
        assert not bad, f"{package} imports {bad}"


def test_model_views_match_registered_protocol():
    assert {k: list(v) for k, v in VARIANTS.items()} == PROTOCOL["variants"]
