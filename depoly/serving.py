
import time, datetime
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form, Query
from .routers import router
from typing import Any
from main import main


class MyRequest(BaseModel):
    imgBase64: str


class MyResponse(BaseModel):
    isSuccess: bool = Field(..., example = True)
    code: int
    message: str = Field(..., example = "success")
    result: Any = Field(...)
    timestamp: str
    timecost: str

@router.post('/kuaidi_rec', summary="kuaidi_rec", response_model=MyResponse)
async def interface(req: MyRequest):
    start = time.time()
    timestamp = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    start = time.time()
    result = main(req.imgBase64)
    end = time.time()
    return MyResponse(isSuccess = True, code=0, message = 'success', result = result,
                      timestamp = timestamp, timecost = f"{((end - start) * 1000):0.0f}ms")

