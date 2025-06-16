import os
import tempfile
from typing import Callable

import ast
from bowler import Query


class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.vars = set()

    def visit_Assign(self, node):
        for target in node.targets:
            self._handle_target(target)

    def visit_AnnAssign(self, node):
        self._handle_target(node.target)

    def visit_AugAssign(self, node):
        self._handle_target(node.target)

    def visit_FunctionDef(self, node):
        params = node.args
        self._collect_params(params)

    def visit_Lambda(self, node):
        self._collect_params(node.args)

    def visit_For(self, node):
        self._handle_target(node.target)

    def visit_With(self, node):
        for item in node.items:
            if item.asname is not None:
                self._handle_target(item.asname.name)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.vars.add(node.name)

    def visit_NamedExpr(self, node):
        self._handle_target(node.target)

    def _collect_params(self, params):
        for param in params.args:
            self.vars.add(param.arg)
        for param in params.posonlyargs:
            self.vars.add(param.arg)
        for param in params.kwonlyargs:
            self.vars.add(param.arg)

    def _handle_target(self, target):
        if isinstance(target, ast.Name):
            self.vars.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._handle_target(element)


def rename_variables(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = ast.parse(code)
    collector = VariableCollector()
    collector.visit(module)

    rename_map = dict()
    for old in collector.vars:
        if not should_apply(old):
            continue
        new = rename_func(old, code)
        if old == new:
            continue
        rename_map[old] = new

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_name = tmp.name
        tmp.flush()

    query = Query(tmp_name)
    for old, new in rename_map.items():
        query = query.select_var(old).rename(new)

    query.execute(write=True, silent=True)

    with open(tmp_name, "r") as f:
        result = f.read()
    os.unlink(tmp_name)

    return result


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class FunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.functions = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not is_dunder(node.name):
            self.functions.add(node.name)
        self.generic_visit(node)


def rename_functions(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = ast.parse(code)
    collector = FunctionCollector()
    collector.visit(module)

    rename_map = dict()
    for old in collector.functions:
        if not should_apply(old):
            continue
        new = rename_func(old, code)
        if old == new:
            continue
        rename_map[old] = new

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_name = tmp.name
        tmp.flush()

    query = Query(tmp_name)
    for old_name, new_name in rename_map.items():
        if old_name != new_name:
            query = query.select_function(old_name).rename(new_name)

    query.execute(write=True, silent=True)

    with open(tmp_name, "r") as f:
        modified_code = f.read()
    os.unlink(tmp_name)

    return modified_code


class ClassCollector(ast.NodeVisitor):
    def __init__(self):
        self.classes = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.add(node.name)
        self.generic_visit(node)


def rename_classes(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = ast.parse(code)
    collector = ClassCollector()
    collector.visit(module)

    rename_map = dict()
    for old in collector.classes:
        if not should_apply(old):
            continue
        new = rename_func(old, code)
        if old == new:
            continue
        rename_map[old] = new

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_name = tmp.name
        tmp.flush()

    query = Query(tmp_name)
    for old_name, new_name in rename_map.items():
        if old_name != new_name:
            query = query.select_class(old_name).rename(new_name)

    query.execute(write=True, silent=True)

    with open(tmp_name, "r") as f:
        modified_code = f.read()
    os.unlink(tmp_name)

    return modified_code
