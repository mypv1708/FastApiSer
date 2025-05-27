import uvicorn
from app.core.config import SAMPLE_RATE

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8386,
        reload=True  # Tự động reload khi có thay đổi code
    ) 