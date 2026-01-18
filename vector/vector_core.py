from raw_data import RawData


class Vector():
    def __init__(self, raw_data: list[RawData]):
        self.raw_data = raw_data

    def macd(self):
        return [1, 1, 1]
