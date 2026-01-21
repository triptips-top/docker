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

    def ma(self, data, period: int):
        ma = numpy.full_like(data, 0)

        for i in range(period-1, len(data)):
            ma[i] = numpy.mean(data[i-period+1:i+1])

        return ma

    def macd(self, data):
        dif = self.ema(data, 12) - self.ema(data, 26)
        dea = self.ema(dif, 9)
        macd = (dif - dea) * 2

        return {"dif": dif, "dea": dea, "macd": macd}


class Vector():
    def __init__(self, raw_data: list[RawData]):
        self.raw_data = raw_data
        self.close = numpy.array([item.收盘 for item in raw_data])

    def kline(self):
        close_60 = self.close[-60:]

        close_min = numpy.min(close_60)
        close_max = numpy.max(close_60)

        vector_close = (close_60-close_min) / (close_max-close_min)

        vector = numpy.asarray(vector_close)

        return vector.tolist()

    def ma(self):
        ma_20 = Ta().ma(self.close, 20)[-40:]
        ma_60 = Ta().ma(self.close, 60)[-40:]
        ma_120 = Ta().ma(self.close, 120)[-40:]

        ma_20_min = numpy.min(ma_20)
        ma_20_max = numpy.max(ma_20)
        ma_60_min = numpy.min(ma_60)
        ma_60_max = numpy.max(ma_60)
        ma_120_min = numpy.min(ma_120)
        ma_120_max = numpy.max(ma_120)

        vector_ma_20 = (ma_20-ma_20_min) / (ma_20_max-ma_20_min)
        vector_ma_60 = (ma_60-ma_60_min) / (ma_60_max-ma_60_min)
        vector_ma_120 = (ma_120-ma_120_min) / (ma_120_max-ma_120_min)

        vector = numpy.column_stack([
            vector_ma_20, vector_ma_60, vector_ma_120
        ]).flatten()

        return vector.tolist()

    def macd(self):
        macd = Ta().macd(self.close)

        dif_20 = macd["dif"][-20:]
        dea_20 = macd["dea"][-20:]
        macd_20 = macd["macd"][-20:]

        dif_abs_max = numpy.max(numpy.abs(dif_20))
        dea_abs_max = numpy.max(numpy.abs(dea_20))
        macd_abs_max = numpy.max(numpy.abs(macd_20))

        vector_dif = dif_20 / dif_abs_max
        vector_dea = dea_20 / dea_abs_max
        vector_macd = macd_20 / macd_abs_max

        vector = numpy.column_stack([
            vector_dif, vector_dea, vector_macd
        ]).flatten()

        return vector.tolist()
