from routes.ask_router import ask
from fastapi import FastAPI
import uvicorn

app = FastAPI()

app.include_router(ask, prefix="/ask")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

