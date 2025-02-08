from fastapi import FastAPI
from routers.routers import router

app = FastAPI(title="Speech-to-Text API")

# Подключаем маршруты
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Speech-to-Text API is running!"}
