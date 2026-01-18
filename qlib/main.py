from fastapi import FastAPI
from rawdata import RawData
from vector import Vector


app = FastAPI()


@app.post("/vector")
def vector(raw_data: list[RawData]):
    vector = Vector(raw_data)

    return {
        "vector_macd": vector.macd()
    }
