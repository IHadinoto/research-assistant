"""
Advanced research agent with multi-tool capabilities.
"""
from typing import List, Dict
import config

# Attempt to import optional LangChain pieces. If the runtime's LangChain
# doesn't expose certain classes we provide safe fallbacks so the module can
# be imported and basic functionality remains available for testing.
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    from langchain.agents import AgentExecutor, create_react_agent
except Exception:
    AgentExecutor = None
    create_react_agent = None

try:
    from langchain.prompts import PromptTemplate
except Exception:
    PromptTemplate = None

try:
    from langchain.chains import RetrievalQA
except Exception:
    RetrievalQA = None

class ResearchAgent:
    """
    Multi-tool research agent with RAG capabilities.
    """
    
    def __init__(self, tools: List, vector_store_manager, memory_manager):
        """
        Initialize research agent.
        
        Args:
            tools: List of LangChain tools
            vector_store_manager: Vector store manager instance
            memory_manager: Memory manager instance
        """
        # Initialize LLM or a simple fallback
        if ChatOllama is not None:
            try:
                self.llm = ChatOllama(
                    model=config.LLM_MODEL,
                    base_url=config.OLLAMA_BASE_URL,
                    temperature=config.TEMPERATURE,
                    num_predict=config.MAX_TOKENS,
                )
            except Exception:
                self.llm = None
        else:
            self.llm = None
        
        self.tools = tools
        self.vector_store = vector_store_manager
        self.memory = memory_manager
        
        # Create RAG chain for document queries (if available)
        if RetrievalQA is not None and self.llm is not None:
            try:
                self.rag_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(),
                    return_source_documents=True,
                )
            except Exception:
                self.rag_chain = None
        else:
            self.rag_chain = None

        # Create agent with tools (use fallback if LangChain agent APIs are missing)
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create ReAct agent with tools."""
        # Advanced ReAct prompt template (use PromptTemplate if available)
        raw_prompt = (
            "You are an advanced research assistant with access to multiple tools and a knowledge base.\n\n"
            "You have access to the following tools:\n{tools}\n\n"
            "Tool Names: {tool_names}\n\n"
            "Use this format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Previous conversation:\n{chat_history}\n\n"
            "Question: {input}\n{agent_scratchpad}\n"
        )

        if PromptTemplate is not None:
            try:
                prompt = PromptTemplate.from_template(raw_prompt)
            except Exception:
                prompt = raw_prompt
        else:
            prompt = raw_prompt

        # If the create_react_agent / AgentExecutor APIs are available use
        # them. Otherwise return a lightweight fallback executor with a
        # predictable interface for testing.
        if create_react_agent is not None and AgentExecutor is not None and PromptTemplate is not None:
            agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=config.MAX_ITERATIONS,
                handle_parsing_errors=True,
                memory=self.memory.memory,
            )
            return agent_executor

        # Fallback simple executor
        class _FallbackExecutor:
            def __init__(self, tools, memory):
                self.tools = tools
                self.memory = memory

            def invoke(self, payload: Dict):
                # Return a clear fallback message so users know functionality
                # is limited in this environment.
                return {"output": "Agent functionality is unavailable in this environment (missing LangChain agent APIs)."}

        return _FallbackExecutor(self.tools, self.memory.memory)
    
    def query_documents(self, question: str) -> Dict:
        """
        Query the document knowledge base using RAG.
        
        Args:
            question: Question to ask about documents
            
        Returns:
            Dictionary with answer and sources
        """
        # If vector store is empty or rag_chain is unavailable, return helpful messages
        try:
            empty_count = self.vector_store.vectorstore._collection.count()
        except Exception:
            empty_count = 0

        if empty_count == 0:
            return {"answer": "No documents loaded in knowledge base.", "sources": []}

        if self.rag_chain is None:
            return {"answer": "RAG chain is not available in this environment.", "sources": []}

        # Use the RAG chain if present
        result = self.rag_chain.invoke({"query": question})

        return {
            "answer": result.get("result", ""),
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
        }
    
    def research(self, query: str) -> str:
        """
        Perform comprehensive research using all available tools.
        
        Args:
            query: Research query
            
        Returns:
            Research results
        """
        try:
            response = self.agent.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"Research error: {str(e)}"
