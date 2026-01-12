from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "ui" / "templates")

@router.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
