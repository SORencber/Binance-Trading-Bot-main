# portfolio/portfolio_manager.py

class PortfolioManager:
    """
    max_risk => portföyün maksimum riske açık oranı (örn. %20)
    initial_eq => ilk sermaye
    max_dd => max drawdown eşiği
    symbol_positions => her sembol için { has_position, quantity, entry_price }
    """
    def __init__(self, max_risk=0.4, initial_eq=10000.0, max_dd=0.15):
        self.max_risk = max_risk
        self.initial_equity = initial_eq
        self.max_drawdown = max_dd
        self.symbol_positions = {}

    def update_position(self, symbol: str, has_position: bool, quantity: float, entry_price: float):
        if symbol not in self.symbol_positions:
            self.symbol_positions[symbol] = {
                "has_position": False,
                "quantity": 0.0,
                "entry_price": 0.0
            }
        self.symbol_positions[symbol]["has_position"] = has_position
        self.symbol_positions[symbol]["quantity"] = quantity
        self.symbol_positions[symbol]["entry_price"] = entry_price

    def check_portfolio_risk(self, symbol: str, proposed_usd: float, total_eq: float) -> bool:
        used = 0
        for sym, pos in self.symbol_positions.items():
            if pos["has_position"]:
                used += pos["quantity"] * pos["entry_price"]
                print(used,pos["has_position"])

        new_ratio = (used + proposed_usd) / total_eq
        print(used,proposed_usd,total_eq,new_ratio,self.max_risk)
        return (new_ratio <= self.max_risk)

    def calc_drawdown(self, current_eq: float) -> float:
        dd = (self.initial_equity - current_eq) / self.initial_equity
        return dd
