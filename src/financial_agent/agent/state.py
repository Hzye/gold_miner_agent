from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class CompanyMetrics(TypedDict):
    """Metrics for a single company"""
    company_name: str
    revenue: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]
    operating_margin: Optional[float]
    profit_margin: Optional[float]
    revenue_growth: Optional[float]


class AgentState(TypedDict):
    """State for the financial analysis agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    pdf_paths: List[str]
    current_pdf_index: int  # Track which PDF we're processing
    current_pdf: Optional[str]
    extracted_text: Optional[str]
    current_company: Optional[str]
    raw_metrics: Optional[Dict]  # Store extracted raw values
    company_metrics: List[CompanyMetrics]
    comparison_csv_path: Optional[str]
    error: Optional[str]