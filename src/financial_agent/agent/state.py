from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class CompanyMetrics(TypedDict):
    """Metrics for single company"""
    company_name: str
    revenue: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]


class AgentState(TypedDict):
    """State for financial agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    pdf_paths: List[str]
    current_pdf: Optional[str]
    extracted_text: Optional[str]
    current_company: Optional[str]
    company_metrics: List[CompanyMetrics]
    comparison_csv_path: Optional[str]
    error: Optional[str]