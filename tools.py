from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def web_search(query: str) -> str:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query.

    Returns:
        The search results.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)
