class OrderbookAnalyzer:
    def evaluate_orderbook(self, row)-> float:
        ask50= row.get("AskVolume_50",0.0)
        bid50= row.get("BidVolume_50",0.0)
        ratio= (ask50+1e-9)/(bid50+1e-9)
        if ratio>2.0:
            return -1
        elif ratio<0.5:
            return +1
        return 0
