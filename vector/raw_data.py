from pydantic import BaseModel


class RawData(BaseModel):
    日期: str
    股票代码: str
    开盘: float
    收盘: float
    最高: float
    最低: float
    成交量: float
    成交额: float
    振幅: float
    涨跌幅: float
    涨跌额: float
    换手率: float
