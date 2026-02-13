from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    parse_pdf_node,
    extract_metrics_node,
    calculate_metrics_node,
    should_process_more_pdfs,
    generate_comparison_node
)


def create_financial_agent():
    """Create langgraph workflow"""

    workflow = StateGraph(AgentState)

    # add nodes
    workflow.add_node("parse_pdf", parse_pdf_node)
    workflow.add_node("extract_metrics", extract_metrics_node)
    workflow.add_node("calculate_metrics", calculate_metrics_node)
    workflow.add_node("generate_comparison", generate_comparison_node)

    # edges
    workflow.set_entry_point("parse_pdf")
    workflow.add_edge("parse_pdf", "extract_metrics")
    workflow.add_edge("extract_metrics", "generate_comparison_node")

    # conditional edge
    workflow.add_conditional_edges(
        "calculate_metrics",
        should_process_more_pdfs,
        {
            "parse_pdf": "parse_pdf",
            "generate_comparison": "generate_comparison"
        }
    )

    workflow.add_edge("generate_comparison", END)

    return workflow.compile()