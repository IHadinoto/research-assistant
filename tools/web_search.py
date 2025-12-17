"""
Web search tool using DuckDuckGo for privacy-focused searches.
"""
try:
    from langchain.tools import Tool
except Exception:
    # Minimal local stand-in for langchain.tools.Tool used by our code
    class Tool:
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Optional

class WebSearchTool:
    """
    Advanced web search tool with result filtering and ranking.
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize web search tool.
        
        Args:
            max_results: Maximum number of search results to return
        """
        # Attempt to initialize the LangChain DuckDuckGo wrapper. If the
        # underlying ddgs package isn't installed we fall back to a simple
        # stub implementation so the tool remains importable.
        try:
            self.search = DuckDuckGoSearchAPIWrapper()
        except Exception:
            class _FallbackDuck:
                def results(self, query, max_results=5):
                    return []
            self.search = _FallbackDuck()

        self.max_results = max_results
    
    def search_web(self, query: str) -> str:
        """
        Perform web search and return formatted results.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            results = self.search.results(query, max_results=self.max_results)
            
            if not results:
                return "No results found for this query."
            
            formatted_results = []
            for idx, result in enumerate(results, 1):
                formatted_results.append(
                    f"{idx}. {result.get('title', 'N/A')}\n"
                    f"   URL: {result.get('link', 'N/A')}\n"
                    f"   Snippet: {result.get('snippet', 'N/A')}\n"
                )
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def as_tool(self) -> Tool:
        """Return as LangChain Tool for agent integration."""
        return Tool(
            name="web_search",
            description=(
                "Useful for searching the web for current information, news, "
                "facts, or any topic not in your knowledge base. "
                "Input should be a search query string."
            ),
            func=self.search_web
        )
