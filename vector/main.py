from fastapi import FastAPI
from raw_data import RawData
from vector_core import Vector


app = FastAPI()


@app.post("/vector")
def vector(raw_data: list[RawData]):
    vector = Vector(raw_data)

    return {
        "vector_macd": vector.macd()
    }
