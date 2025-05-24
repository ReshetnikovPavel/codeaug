import os
import tempfile
from typing import Callable

import libcst as cst
from bowler import Query


class VariableCollector(cst.CSTVisitor):
    def __init__(self):
        self.vars = set()

    def visit_Assign(self, node):
        for target in node.targets:
            self._handle_target(target.target)

    def visit_AnnAssign(self, node):
        self._handle_target(node.target)

    def visit_AugAssign(self, node):
        self._handle_target(node.target)

    def visit_FunctionDef(self, node):
        params = node.params
        self._collect_params(params)

    def visit_Lambda(self, node):
        self._collect_params(node.params)

    def visit_For(self, node):
        self._handle_target(node.target)

    def visit_With(self, node):
        for item in node.items:
            if item.asname is not None:
                self._handle_target(item.asname.name)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.vars.add(node.name.value)

    def visit_NamedExpr(self, node):
        self._handle_target(node.target)

    def _collect_params(self, params):
        for param in params.params:
            self.vars.add(param.name.value)
        for param in params.posonly_params:
            self.vars.add(param.name.value)
        for param in params.kwonly_params:
            self.vars.add(param.name.value)
        if params.star_arg:
            self.vars.add(params.star_arg.name)
        if params.star_kwarg:
            self.vars.add(params.star_kwarg.name)

    def _handle_target(self, target):
        if isinstance(target, cst.Name):
            self.vars.add(target.value)
        elif isinstance(target, (cst.Tuple, cst.List)):
            for element in target.elements:
                self._handle_target(element.value)


def rename_variables(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = cst.parse_module(code)
    collector = VariableCollector()
    module.visit(collector)
    rename_map = {old: rename_func(old, code) for old in collector.vars if should_apply(old)}

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


class FunctionCollector(cst.CSTVisitor):
    def __init__(self):
        self.functions = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.functions.add(node.name.value)


def rename_functions(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = cst.parse_module(code)
    collector = FunctionCollector()
    module.visit(collector)
    rename_map = {
        old: rename_func(old, code) for old in collector.functions if should_apply(old)
    }

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


class ClassCollector(cst.CSTVisitor):
    def __init__(self):
        self.classes = set()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.classes.add(node.name.value)


def rename_classes(
    code: str,
    rename_func: Callable[[str, str], str],
    should_apply: Callable[[str], bool] = lambda _: True,
) -> str:
    module = cst.parse_module(code)
    collector = ClassCollector()
    module.visit(collector)
    rename_map = {
        old: rename_func(old, code) for old in collector.classes if should_apply(old)
    }

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
