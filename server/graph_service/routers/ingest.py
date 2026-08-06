import asyncio
import logging
import os
import random
from functools import partial

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep


# Septics Hub patch: run multiple worker coroutines so we can overlap
# the slow OpenAI extract+embed step (~10 s/episode) across the queue.
# A single worker caps throughput at ~6 episodes/min; 4 workers raise
# that ceiling to ~24 episodes/min, which is the difference between a
# production seed completing in 6 h vs ~45 min. Configurable via env
# var GRAPHITI_WORKER_CONCURRENCY (defaults to 4) so the deployment
# can dial it up/down depending on the OpenAI rate-limit budget.
_DEFAULT_WORKER_CONCURRENCY = 4
_DEFAULT_WORKER_MAX_RETRIES = 3


def _resolve_worker_concurrency() -> int:
    raw = os.environ.get('GRAPHITI_WORKER_CONCURRENCY')
    if not raw:
        return _DEFAULT_WORKER_CONCURRENCY
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_WORKER_CONCURRENCY
    return max(1, min(n, 16))


def _resolve_worker_max_retries() -> int:
    raw = os.environ.get('GRAPHITI_WORKER_MAX_RETRIES')
    if not raw:
        return _DEFAULT_WORKER_MAX_RETRIES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_WORKER_MAX_RETRIES
    # Clamp: 1 = no retries, 10 = 9 retries. Beyond that a stuck message can
    # tie up a worker for minutes.
    return max(1, min(n, 10))


# Septics Hub patch: import neo4j transient error classes lazily so the module
# still imports cleanly on machines that don't have neo4j installed (tests).
# We treat these as retryable; everything else is treated as permanent.
def _neo4j_transient_exceptions() -> tuple[type[BaseException], ...]:
    try:
        from neo4j.exceptions import (  # type: ignore
            ServiceUnavailable,
            SessionExpired,
            TransientError,
            WriteServiceUnavailable,
        )
    except ImportError:
        return ()
    return (ServiceUnavailable, SessionExpired, TransientError, WriteServiceUnavailable)


def _is_transient_neo4j_error(exc: BaseException) -> bool:
    transient_types = _neo4j_transient_exceptions()
    if transient_types and isinstance(exc, transient_types):
        return True
    # Fallback for driver messages we've seen in production even when the
    # exception type check fails (e.g. because graphiti wraps it in another
    # exception): pattern-match the message.
    msg = str(exc)
    return (
        'SessionExpired' in msg
        or 'ServiceUnavailable' in msg
        or 'Failed to obtain connection' in msg
        or 'connection is closed' in msg.lower()
    )


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.concurrency = _resolve_worker_concurrency()
        self.max_retries = _resolve_worker_max_retries()

    async def worker(self, worker_id: int):
        # Septics Hub patch: catch ALL exceptions from a job so one failure
        # cannot kill the worker. Upstream only catches CancelledError, which
        # means any OpenAI/Neo4j/payload error permanently stops processing
        # while POST /messages keeps returning 202 (issue #566).
        while True:
            try:
                job = await self.queue.get()
            except asyncio.CancelledError:
                break
            print(f'Worker {worker_id} got a job: (size of remaining queue: {self.queue.qsize()})')
            try:
                await self._run_with_retries(worker_id, job)
            except asyncio.CancelledError:
                break
            finally:
                self.queue.task_done()

    async def _run_with_retries(self, worker_id: int, job) -> None:
        # Septics Hub patch: retry on transient Neo4j errors before dropping.
        # SessionExpired / ServiceUnavailable are common after a Neo4j restart
        # and were producing silent data loss (whole batch dropped). We back
        # off with jitter so a Neo4j restart storm doesn't stampede.
        attempt = 0
        while True:
            attempt += 1
            try:
                await job()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - we log below
                transient = _is_transient_neo4j_error(exc)
                if transient and attempt < self.max_retries:
                    delay = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.5)
                    logging.warning(
                        '[graphiti] worker %s: transient Neo4j error on attempt %s/%s, '
                        'retrying in %.1fs: %s',
                        worker_id,
                        attempt,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        raise
                    continue
                if transient:
                    logging.error(
                        '[graphiti] worker %s: transient Neo4j error exhausted %s retries; '
                        'dropping job for hub cron to reingest: %s',
                        worker_id,
                        self.max_retries,
                        exc,
                    )
                else:
                    logging.exception(
                        '[graphiti] ingest job failed on worker %s; worker continues',
                        worker_id,
                    )
                return

    async def start(self):
        self.tasks = [
            asyncio.create_task(self.worker(i + 1))
            for i in range(self.concurrency)
        ]
        print(
            f'AsyncWorker started with concurrency={self.concurrency}, '
            f'max_retries={self.max_retries}'
        )

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.tasks = []
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


# Septics Hub patch: worker start/stop is now driven from the app lifespan
# in main.py. The router no longer carries its own lifespan, which avoids
# any reliance on FastAPI nested-lifespan merging.
router = APIRouter()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_messages_task(m: Message):
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
