from fastapi import FastAPI
from pydantic import BaseModel
import qlib


class Item(BaseModel):
    id: int
    create_at: str
    update_at: str
    symbol: str
    entry_time: str
    raw_data: list
    vector_macd: None


app = FastAPI()


@app.post("/vector")
def vector(item: Item):
    item.vector_macd = [1, 1, 1]

    return {
        "id": item.id,
        "vector_macd": item.vector_macd,
        "qlib": qlib.__version__
    }
