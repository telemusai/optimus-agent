"""Project memory operations. Existing session/global CRUD remains in rlm.harness."""
from rlm import host_request

async def request(action: str, **payload):
    """Run a memory operation; retain the origin label when displaying recalled data."""
    return await host_request("memory.request", {"action": action, **payload})

async def status():
    return await request("status")

async def search(query: str = "", *, include_inactive: bool = False, scope: str | None = None):
    return await request("search", query=query, includeInactive=include_inactive, **({"scope": scope} if scope else {}))

async def read(id: str):
    return await request("read", id=id)

async def apply(proposal: dict, *, event_id: str, revision: int, sources=None, host=False, automatic=False):
    return await request("apply", proposal=proposal, eventId=event_id, revision=revision, sources=sources or [], host=host, automatic=automatic)

async def handoff(task: str, state: str, decisions: str, unresolved: str, *, event_id: str, revision: int, sources=None, automatic=False):
    return await request("handoff", task=task, state=state, decisions=decisions, unresolved=unresolved, eventId=event_id, revision=revision, sources=sources or [], automatic=automatic)

async def source(path: str):
    return await request("source", path=path)

async def configure(**settings):
    return await request("configure", settings=settings)

async def import_prepare(path: str):
    return await request("import_prepare", path=path)

async def import_read(id: str):
    return await request("import_read", id=id)

async def import_run(id: str):
    return await request("import_run", id=id)

async def import_chunk(id: str, chunk: int):
    return await request("import_chunk", id=id, chunk=chunk)

async def import_apply(id: str, *, revision: int):
    return await request("import_apply", id=id, revision=revision)

async def share(ids: list[str], *, remove=None):
    return await request("share", ids=ids, remove=remove or [])

async def sync():
    return await request("sync")

async def backup():
    return await request("backup")

async def restore(id: str):
    return await request("restore", id=id)

async def history():
    return await request("history")

async def rollback(id: str, *, revision: int):
    return await request("rollback", id=id, revision=revision)
