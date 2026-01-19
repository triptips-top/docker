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

        dif_abs_max = numpy.max(numpy.abs(macd["dif"][-20:]))
        dea_abs_max = numpy.max(numpy.abs(macd["dea"][-20:]))
        macd_abs_max = numpy.max(numpy.abs(macd["macd"][-20:]))

        vector_dif = macd["dif"][-20:] / dif_abs_max
        vector_dea = macd["dea"][-20:] / dea_abs_max
        vector_macd = macd["macd"][-20:] / macd_abs_max

        vector = numpy.column_stack([
            vector_dif, vector_dea, vector_macd
        ]).flatten()

        return vector.tolist()
