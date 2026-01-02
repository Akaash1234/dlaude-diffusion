# api for dlaude chat interface
import asyncio, json, uuid
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from .model import get_model, TinyDiffusionModel

# in-memory storage (good enough for now)
conversations: Dict[str, List[dict]] = {}

class MessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    max_tokens: int = 512
    temperature: float = 0.8  # dont use 0, causes div by zero lol
    steps: int = 32

class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int

model_status = {"loading": False, "loaded": False, "progress": 0, "message": "Not started", "error": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_model_async())
    yield

app = FastAPI(title="Dlaude Chat", version="1.0.0", lifespan=lifespan)

async def load_model_async():
    global model_status
    model_status["loading"] = True
    model_status["message"] = "Starting..."
    def cb(p, msg):
        model_status["progress"] = p
        model_status["message"] = msg
    try:
        model = get_model()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.load_model(cb))
        model_status["loaded"] = True
        model_status["loading"] = False
    except Exception as e:
        model_status["loading"] = False
        model_status["error"] = str(e)
        model_status["message"] = f"Failed: {e}"

@app.get("/health")
async def health(): return {"status": "ok", "model_status": model_status}

@app.get("/api/model/status")
async def get_model_status(): return model_status

@app.get("/api/conversations")
async def list_convos() -> List[ConversationInfo]:
    result = []
    for cid, msgs in conversations.items():
        if not msgs: continue
        title = "New Chat"
        for m in msgs:
            if m["role"] == "user":
                title = m["content"][:50] + ("..." if len(m["content"]) > 50 else "")
                break
        result.append(ConversationInfo(id=cid, title=title, created_at=msgs[0].get("timestamp", ""), message_count=len(msgs)))
    return sorted(result, key=lambda x: x.created_at, reverse=True)

@app.post("/api/conversations")
async def create_convo(): 
    cid = str(uuid.uuid4())
    conversations[cid] = []
    return {"id": cid}

@app.get("/api/conversations/{cid}")
async def get_convo(cid: str):
    if cid not in conversations: raise HTTPException(status_code=404, detail="Not found")
    return {"id": cid, "messages": conversations[cid]}

@app.delete("/api/conversations/{cid}")
async def del_convo(cid: str):
    if cid in conversations: del conversations[cid]
    return {"status": "deleted"}

@app.post("/api/chat")
async def chat(req: MessageRequest):
    if not model_status["loaded"]:
        if model_status["loading"]: raise HTTPException(status_code=503, detail="Model loading")
        raise HTTPException(status_code=503, detail=f"Model failed: {model_status.get('error')}")
    
    cid = req.conversation_id or str(uuid.uuid4())
    if cid not in conversations: conversations[cid] = []
    
    # save user msg
    conversations[cid].append({
        "role": "user", 
        "content": req.message, 
        "timestamp": datetime.now().isoformat()
    })
    
    async def stream():
        model = get_model()
        yield {"event": "start", "data": json.dumps({"conversation_id": cid})}
        
        # queue for thread -> async communication
        q = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def step_cb(cur, tot, txt):
            loop.call_soon_threadsafe(q.put_nowait, {"type": "progress", "step": cur, "total": tot, "text": txt})
        
        async def producer():
            try:
                chunks = []
                def run():
                    for chunk in model.generate(prompt=req.message, max_tokens=req.max_tokens, 
                                                temperature=req.temperature, steps=req.steps, step_callback=step_cb):
                        chunks.append(chunk)
                await loop.run_in_executor(None, run)
                loop.call_soon_threadsafe(q.put_nowait, {"type": "done", "text": "".join(chunks)})
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "error": str(e)})
        
        task = asyncio.create_task(producer())
        full_resp = ""
        
        while True:
            item = await q.get()
            if item["type"] == "progress":
                yield {"event": "diffusing", "data": json.dumps({"step": item["step"], "total_steps": item["total"], "text": item["text"]})}
            elif item["type"] == "done":
                full_resp = item["text"]
                conversations[cid].append({"role": "assistant", "content": full_resp, "timestamp": datetime.now().isoformat()})
                yield {"event": "done", "data": json.dumps({"conversation_id": cid, "full_response": full_resp})}
                break
            elif item["type"] == "error":
                yield {"event": "error", "data": json.dumps({"error": item["error"]})}
                break
        await task
    
    return EventSourceResponse(stream())

# static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(): return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
