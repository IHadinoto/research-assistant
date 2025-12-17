"""
Calculator tool for mathematical operations.
"""
try:
    from langchain.tools import Tool
except Exception:
    class Tool:
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func

import re

class CalculatorTool:
    """
    Safe mathematical calculator for agent use.
    """
    
    def calculate(self, expression: str) -> str:
        """
        Safely evaluate mathematical expressions.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of calculation or error message
        """
        try:
            # Remove any non-mathematical characters for safety
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Evaluate expression
            result = eval(safe_expr, {"__builtins__": {}}, {})
            return f"Result: {result}"
        
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def as_tool(self) -> Tool:
        """Return as LangChain Tool for agent integration."""
        return Tool(
            name="calculator",
            description=(
                "Useful for performing mathematical calculations. "
                "Input should be a mathematical expression like '25 * 4' or '(100 + 50) / 2'."
            ),
            func=self.calculate
        )
