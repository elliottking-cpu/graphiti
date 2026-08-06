import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.routers.ingest import async_worker
from graph_service.zep_graphiti import initialize_graphiti

logger = logging.getLogger(__name__)


# Septics Hub patch: deep healthcheck timeout. Kept short so that Railway /
# hub watchdog probes don't stall the request thread when Neo4j is wedged.
DEEP_HEALTHCHECK_TIMEOUT_SECONDS = 2.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Septics Hub patch: build ONE ZepGraphiti (and therefore one Neo4j
    # driver) for the lifetime of the process and stash it on app.state.
    # Previously the DI created + closed a client per request, which meant
    # queued AsyncWorker jobs were racing driver teardown.
    app.state.graphiti = await initialize_graphiti(settings)
    await async_worker.start()
    try:
        yield
    finally:
        await async_worker.stop()
        try:
            await app.state.graphiti.close()
        except Exception:
            logger.exception('[graphiti] error closing Graphiti client on shutdown')


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck/live')
async def healthcheck_live():
    """Liveness probe: is the HTTP process alive? Used by Railway container
    healthcheck. Does NOT touch Neo4j so a database blip cannot trigger an
    infinite Railway restart loop."""
    return JSONResponse(content={'status': 'live'}, status_code=200)


@app.get('/healthcheck')
async def healthcheck():
    """Deep readiness probe: verifies we can round-trip a query to Neo4j.
    Returns 503 when Neo4j is unreachable so the hub watchdog can trigger a
    Railway redeploy via serviceInstanceRedeploy."""
    client = getattr(app.state, 'graphiti', None)
    if client is None:
        return JSONResponse(
            content={'status': 'starting', 'detail': 'Graphiti not yet initialised'},
            status_code=503,
        )
    try:
        await asyncio.wait_for(
            client.driver.execute_query('RETURN 1 AS ok'),
            timeout=DEEP_HEALTHCHECK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            content={
                'status': 'unhealthy',
                'detail': f'Neo4j did not respond within {DEEP_HEALTHCHECK_TIMEOUT_SECONDS}s',
            },
            status_code=503,
        )
    except Exception as exc:
        return JSONResponse(
            content={'status': 'unhealthy', 'detail': f'Neo4j error: {exc}'},
            status_code=503,
        )
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
