from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .runtime import SimulationRuntime


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>embodi simulation</title><style>
body{margin:0;background:#0b0e13;color:#e9eef6;font:15px ui-monospace,monospace;display:grid;place-items:center;min-height:100vh}
main{width:min(94vw,1000px)}h1{font-size:18px;font-weight:500;letter-spacing:.08em}
img{display:block;width:100%;border:1px solid #293140;background:#111;border-radius:8px}
footer{display:flex;align-items:center;gap:16px;margin-top:12px}button{background:#e9eef6;border:0;border-radius:5px;padding:10px 18px;cursor:pointer}
#status{color:#9eabbc}
</style></head><body><main><h1>embodi / so101 pick-place / mujoco</h1>
<img src="/stream.mjpg"><footer><button onclick="resetEpisode()">reset episode</button><span id="status">connecting</span></footer>
</main><script>
async function resetEpisode(){document.querySelector('button').disabled=true;await fetch('/api/reset',{method:'POST'});document.querySelector('button').disabled=false}
async function poll(){try{let r=await fetch('/api/status');let s=await r.json();document.querySelector('#status').textContent=`episode ${s.episode} · ${s.sim_time}s · ${s.success?'success':s.lifted?'lifted':'running'} · inference ${s.inference_ms}ms`}catch(e){}setTimeout(poll,500)}poll()
</script></body></html>"""


def create_app(runtime: SimulationRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        yield
        runtime.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE

    @app.get("/api/status")
    async def status():
        state = runtime.status()
        if runtime.error is not None:
            raise HTTPException(503, str(runtime.error))
        return state

    @app.post("/api/reset")
    async def reset():
        try:
            return await asyncio.to_thread(runtime.request_reset)
        except (RuntimeError, TimeoutError) as error:
            raise HTTPException(503, str(error)) from error

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    async def frames():
        sequence = -1
        while True:
            next_sequence, jpeg = runtime.frame()
            if jpeg and next_sequence != sequence:
                sequence = next_sequence
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.02)

    @app.get("/stream.mjpg")
    async def stream():
        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"},
        )

    return app
