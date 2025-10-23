from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from services import (
    IndexingService
)
from config import Config

@asynccontextmanager
async def lifespan_event(app: FastAPI):

    config = Config()

    app.state.indexing_service = IndexingService(config)

    yield

class Boostrapper:

    def run(self) -> FastAPI:

        app = FastAPI(lifespan=lifespan_event,
                      title="MultiVector RAG API",
                      version="1.0",
                      summary="API to test multirag vector with 3 differents scenario")
     
        # Global exception handler for any unhandled exceptions
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            #request.app.state.logger.error("Unhandled exception occurred", exc_info=exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )


        return app
     
