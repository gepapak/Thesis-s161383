# environment_financials.py
# Financial helper functions extracted from environment.py.

from __future__ import annotations

from typing import Any, Dict


class FinancialManager:
    """Handles simple NAV, MTM, and transaction-cost helpers."""

    def __init__(self, config: Any, financial_engine: Any = None):
        self.config = config
        self.financial_engine = financial_engine

    def calculate_nav(self, positions: Dict[str, float], prices: Dict[str, float]) -> float:
        nav = 0.0
        for asset, quantity in positions.items():
            if asset in prices:
                nav += float(quantity) * float(prices[asset])
        return float(nav)

    def calculate_mtm(self, positions: Dict[str, float], prices: Dict[str, float]) -> float:
        return self.calculate_nav(positions, prices)

    def get_market_prices(self, timestep: int) -> Dict[str, float]:
        _ = timestep
        return {}

    def calculate_pnl(self, positions: Dict[str, float], price_changes: Dict[str, float]) -> float:
        pnl = 0.0
        for asset, change in price_changes.items():
            if asset in positions:
                pnl += float(positions[asset]) * float(change)
        return float(pnl)

    def apply_transaction_costs(self, trade_volume: float, price: float) -> float:
        rate = float(getattr(self.config, "trading_cost_rate", 0.0))
        return float(trade_volume) * float(price) * rate

    def validate_financial_constraints(self, positions: Dict[str, float]) -> bool:
        _ = positions
        return True
