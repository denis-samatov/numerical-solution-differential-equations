import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_demo(monkeypatch, step=0.2):
    monkeypatch.setattr(plt, "show", lambda: None)
    script = Path(__file__).parents[1] / "solve_DE.py"
    tree = ast.parse(script.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "h" for t in node.targets):
            node.value = ast.Constant(step)
    ns = {"__name__": "__main__"}
    try:
        with redirect_stdout(StringIO()):
            exec(compile(ast.fix_missing_locations(tree), str(script), "exec"), ns)
    finally:
        plt.close("all")
    return ns


def test_error_reference_keeps_full_precision(monkeypatch):
    ns = load_demo(monkeypatch)
    np.testing.assert_allclose(ns["ans_rightDecision"], ns["exact_solution"](ns["t_list"]), rtol=0, atol=1e-14)


def test_adams_bootstrap_is_fully_numerical(monkeypatch):
    ns = load_demo(monkeypatch)
    np.testing.assert_array_equal(ns["x"][:4], ns["ans_rungeKuttaMethod"][:4])


def test_non_dividing_step_has_consistent_grid(monkeypatch):
    ns = load_demo(monkeypatch, 0.3)
    assert len(ns["t_list"]) == len(ns["ans_rungeKuttaMethod"])
    assert ns["t_list"][-1] <= ns["tn"]
