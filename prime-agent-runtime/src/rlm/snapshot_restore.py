"""Prepare restored callables before committing the live kernel namespace."""

from __future__ import annotations

import functools
import types
from typing import Any

ALWAYS_SKIP = {"rlm", "mcp", "bash", "asyncio", "In", "Out", "get_ipython", "exit", "quit", "open"}
RESTORE_SKIP = {"In", "Out", "get_ipython"}
_ATOMS = (int, float, str, bytes, bool, type(None))


def _replace_tuple_references(value: Any, old: tuple[Any, ...], new: tuple[Any, ...]) -> None:
    """Backpatch direct old-tuple references in an already revived graph."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            if item is old:
                value[index] = new
        return
    if isinstance(value, dict):
        for key, item in list(value.items()):
            replacement_key = new if key is old else key
            if key is old:
                del value[key]
            value[replacement_key] = new if item is old else item
        return
    if isinstance(value, functools.partial):
        args = new if value.args is old else tuple(new if arg is old else arg for arg in value.args)
        value.__setstate__((
            value.func,
            args,
            {key: new if arg is old else arg for key, arg in value.keywords.items()},
            {key: new if attr is old else attr for key, attr in value.__dict__.items()},
        ))
        return
    if isinstance(value, types.FunctionType):
        if value.__defaults__ is old:
            value.__defaults__ = new
        elif value.__defaults__:
            value.__defaults__ = tuple(new if item is old else item for item in value.__defaults__)
        if value.__kwdefaults__:
            value.__kwdefaults__ = {
                key: new if item is old else item for key, item in value.__kwdefaults__.items()
            }
        value.__dict__.update({key: new if item is old else item for key, item in value.__dict__.items()})
        for cell in value.__closure__ or ():
            try:
                if cell.cell_contents is old:
                    cell.cell_contents = new
            except ValueError:
                pass


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
        # Dict backpatching may replace a rebuilt tuple key during revival.
        # Iterate over a snapshot so that legal key replacement cannot mutate
        # the live iterator and drop the entire restored dictionary.
        for key, item in enumerate(value) if isinstance(value, list) else list(value.items()):
            revived = revive(item)
            if revived is not item:
                value[key] = revived
        return value
    if type(value) is tuple:
        # Seed before descent: tuple -> list/dict -> tuple cycles otherwise
        # recurse forever. The original is only a temporary placeholder; after
        # rebuilding (usually because a function changed), rebind every direct
        # reference held by already-revived mutable objects to the new tuple.
        memo[id(value)] = value
        items = tuple(revive(item) for item in value)
        rebuilt = value if all(new is old for new, old in zip(items, value)) else items
        memo[id(value)] = rebuilt
        if rebuilt is not value:
            for dependent in list(memo.values()):
                _replace_tuple_references(dependent, value, rebuilt)
        return rebuilt
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
