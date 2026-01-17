class n8nQlib:
    def __init__(self, items: list):
        self.item = items[0]["json"]

    def vector_macd(self) -> list:
        return [{"json":  {
            "id": self.item["id"],
            "vector_macd": [1, 1, 1]
        }}]


if __name__ == "__main__":
    pass
