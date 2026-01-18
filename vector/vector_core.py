import numpy
from raw_data import RawData


class Ta():
    def ema(self, data, period: int):
        alpha = 2 / (period+1)
        ema = numpy.full_like(data, 0)

        ema[period-1] = numpy.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1-alpha) * ema[i-1]

        return ema

    def macd(self, data):
        dif = self.ema(data, 12) - self.ema(data, 26)
        dea = self.ema(dif, 9)
        macd = (dif - dea) * 2

        return {"dif": dif, "dea": dea, "macd": macd}


class Vector():
    def __init__(self, raw_data: list[RawData]):
        self.raw_data = raw_data
        self.close = numpy.array([item.收盘 for item in raw_data])

    def macd(self):
        macd = Ta().macd(self.close)

        vector_dif = macd["dif"] / numpy.max(numpy.abs(macd["dif"]))
        vector_dea = macd["dea"] / numpy.max(numpy.abs(macd["dea"]))
        vector_macd = macd["macd"] / numpy.max(numpy.abs(macd["macd"]))

        vector = numpy.column_stack([
            vector_dif[-100:], vector_dea[-100:], vector_macd[-100:]
        ]).flatten()

        return vector.tolist()
