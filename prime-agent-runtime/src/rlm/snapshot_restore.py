"""Prepare restored callables before committing the live kernel namespace."""

from __future__ import annotations

import functools
import types
from typing import Any

ALWAYS_SKIP = {"rlm", "mcp", "bash", "asyncio", "In", "Out", "get_ipython", "exit", "quit", "open"}
RESTORE_SKIP = {"In", "Out", "get_ipython"}
_ATOMS = (int, float, str, bytes, bool, type(None))


def _revive(
    value: Any,
    ns: dict[str, Any],
    backfill: list[tuple[str, Any]],
    skip: set[str],
    memo: dict[int, Any],
) -> Any:
    if value is ns:  # dill can restore globals() by reference; never walk live state.
        return value
    if id(value) in memo:
        return memo[id(value)]

    def revive(dep: Any) -> Any:
        return dep if type(dep) in _ATOMS else _revive(dep, ns, backfill, skip, memo)

    if isinstance(value, functools.partial):
        # A partial can cycle through its args or attributes without a function.
        rebuilt = functools.partial(value.func)
        memo[id(value)] = rebuilt
        rebuilt.__setstate__((
            revive(value.func),
            tuple(revive(arg) for arg in value.args),
            {key: revive(arg) for key, arg in value.keywords.items()},
            {key: revive(attr) for key, attr in value.__dict__.items()},
        ))
        return rebuilt
    if isinstance(value, (list, dict)):
        memo[id(value)] = value
        for key, item in enumerate(value) if isinstance(value, list) else value.items():
            revived = revive(item)
            if revived is not item:
                value[key] = revived
        return value
    if type(value) is tuple:
        items = tuple(revive(item) for item in value)
        if all(new is old for new, old in zip(items, value)):
            items = value
        return memo.setdefault(id(value), items)
    if not isinstance(value, types.FunctionType) or value.__module__ != "__main__":
        return value

    rebound = types.FunctionType(value.__code__, ns, value.__name__, None, value.__closure__)
    memo[id(value)] = rebound
    for name, dep in value.__globals__.items():
        if name not in ns and not name.startswith("_") and name not in skip:
            backfill.append((name, revive(dep)))
    if value.__defaults__:
        rebound.__defaults__ = tuple(revive(dep) for dep in value.__defaults__)
    if value.__kwdefaults__:
        rebound.__kwdefaults__ = {key: revive(dep) for key, dep in value.__kwdefaults__.items()}
    # Keep closure cells shared with sibling functions, including attribute-held siblings.
    for cell in value.__closure__ or ():
        if id(cell) in memo:
            continue
        memo[id(cell)] = cell
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        cell.cell_contents = revive(contents)
    rebound.__doc__ = value.__doc__
    rebound.__dict__.update({key: revive(attr) for key, attr in value.__dict__.items()})
    rebound.__annotations__ = value.__annotations__
    rebound.__qualname__ = value.__qualname__
    rebound.__module__ = value.__module__
    params = getattr(value, "__type_params__", None)
    if params is not None:
        rebound.__type_params__ = params
    return rebound


def prepare_restored_values(
    staged: dict[str, Any], ns: dict[str, Any], restore_skip: set[str]
) -> tuple[dict[str, Any], list[tuple[str, Any]], list[dict[str, str]]]:
    prepared: dict[str, Any] = {}
    backfill: list[tuple[str, Any]] = []
    failed: list[dict[str, str]] = []
    skip = ALWAYS_SKIP | RESTORE_SKIP | restore_skip
    for name, value in staged.items():
        additions: list[tuple[str, Any]] = []
        try:
            prepared[name] = _revive(value, ns, additions, skip, {})
        except Exception as error:
            try:
                detail = str(error)
            except BaseException:
                detail = "<exception str() failed>"
            failed.append({"name": name, "reason": f"{type(error).__name__}: {detail[:200]}"})
        else:
            backfill.extend(additions)
    return prepared, backfill, failed
