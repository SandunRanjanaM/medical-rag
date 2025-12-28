
from fastapi import FastAPI
from router import router

app = FastAPI()


# Include the router from router.py
app.include_router(router)
