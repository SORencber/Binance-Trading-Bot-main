class OnChainCollector:
    def evaluate_onchain(self, row)-> float:
        nf= row.get("onchain_flow",0.0)
        whale= row.get("whaleAlert",0.0)
        sc=0
        if nf< -2000: sc-=2
        if whale>1000: sc+=1
        return sc