from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    destination = request.url.replace(request.base_url, "")
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=request.method,
            url=destination,
            headers={key: value for (key, value) in request.headers.items() if key != "Host"},
            data=await request.body(),
            cookies=request.cookies
        )
    headers = [(key, value) for (key, value) in response.headers.items() if key != "Transfer-Encoding"]
    return response.content, response.status_code, headers 