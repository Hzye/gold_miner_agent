from typing import Optional
from langchain_core.tools import tool


@tool
def calculate_profit_margin(net_income: float, revenue: float) -> float:
    """Calculate profit margin percentage.
    
    Args:
        net_income: Net income value
        revenue: Total revenue value
    
    Returns:
        Profit margin as percentage
    """
    if revenue == 0:
        return 0.0
    return (net_income / revenue) * 100


@tool
def calculate_operating_margin(operating_income: float, revenue: float) -> float:
    """Calculate operating margin percentage.
    
    Args:
        operating_income: Operating income value
        revenue: Total revenue value
    
    Returns:
        Operating margin as percentage
    """
    if revenue == 0:
        return 0.0
    return (operating_income / revenue) * 100


@tool
def calculate_eps(net_income: float, shares_outstanding: float) -> float:
    """Calculate earnings per share.
    
    Args:
        net_income: Net income value
        shares_outstanding: Number of shares outstanding
    
    Returns:
        Earnings per share
    """
    if shares_outstanding == 0:
        return 0.0
    return net_income / shares_outstanding


@tool
def calculate_revenue_growth(current_revenue: float, previous_revenue: float) -> float:
    """Calculate revenue growth percentage.
    
    Args:
        current_revenue: Current period revenue
        previous_revenue: Previous period revenue
    
    Returns:
        Revenue growth percentage
    """
    if previous_revenue == 0:
        return 0.0
    return ((current_revenue - previous_revenue) / previous_revenue) * 100


# List of all calculation tools
CALCULATION_TOOLS = [
    calculate_profit_margin,
    calculate_operating_margin,
    calculate_eps,
    calculate_revenue_growth,
]