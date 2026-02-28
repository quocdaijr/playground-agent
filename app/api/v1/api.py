from fastapi import APIRouter
from app.api.v1.chatbot import router as chatbot_router

api_router = APIRouter()
api_router.include_router(chatbot_router)
