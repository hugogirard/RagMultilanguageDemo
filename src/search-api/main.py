from fastapi import FastAPI, HTTPException
from config import Config
from request import TroubleInformation
from fastapi.responses import RedirectResponse
from models import CarFix
from typing import List
from services import CarFixService
import logging
import sys

# Configure logger
logger = logging.getLogger('carfixapi')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

car_fix_service = CarFixService()

app = FastAPI(title="Machine Troubleshooting API",
              version="1.0")

@app.post('/api/car/fix')
async def get_car_fix(trouble_information:TroubleInformation) -> List[CarFix]:
    try:
        return await car_fix_service.get_car_fix(trouble_information.brand,
                                                 trouble_information.model,
                                                 trouble_information.fault)
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail='Internal Server Error')
    
@app.get('/', include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")      