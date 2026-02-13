from typing import Optional
from langchain_core.tools import tool

@tool
def calculate_profit_margin(
    net_income: float,
    revenue: float
) -> float:
    """
    Calculate profit margin percentage.

    Args:
        net_income: Net income value
        revenue: Total revenue value
    
    Returns:
        Profit margin as %
    """
    if revenue == 0:
        return 0.0
    return (net_income / revenue) * 100
