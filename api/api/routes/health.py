from fastapi import APIRouter

health_route = APIRouter(prefix="/api/v1", tags=["api_v1"])


@health_route.get("/health")
async def health_check():
    return {"status": "healthy"}
