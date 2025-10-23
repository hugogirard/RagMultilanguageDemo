from fastapi import FastAPI
from bootstrapper import Boostrapper
from routes import routes
from fastapi.responses import RedirectResponse

app = Boostrapper().run()

for route in routes:
    app.include_router(route,prefix="/api")

@app.get('/', include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")    