import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Literal
from main import run_agent, clear_memory, get_initial_weather

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    type: Literal["weather", "news", "both", "chat", "rejection"]
    message: str
    city: Optional[str] = None
    temperature: Optional[float] = None
    description: Optional[str] = None
    articles: Optional[List[str]] = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    clear_memory()
    return templates.TemplateResponse(request, "index.html")


@app.get("/initial-weather")
def initial_weather():
    """Real weather for the dashboard as soon as the page opens."""
    return get_initial_weather("Faisalabad")


@app.post("/chat")
async def chat(request: ChatRequest):
    result = await run_agent(request.message)
    return ChatResponse(**result)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)