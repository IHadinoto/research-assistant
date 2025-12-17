"""
Main application entry point for the Advanced Research Assistant.
"""
from embeddings.vector_store import VectorStoreManager
from agents.memory_manager import MemoryManager
from agents.research_agent import ResearchAgent
from tools.web_search import WebSearchTool
from tools.calculator import CalculatorTool
import config
from pathlib import Path

class ResearchAssistant:
    """
    Main application class orchestrating all components.
    """
    
    def __init__(self):
        """Initialize all components."""
        print("🚀 Initializing Advanced Research Assistant...")
        
        # Initialize vector store
        print("📚 Loading vector store...")
        self.vector_store = VectorStoreManager()
        
        # Initialize memory
        print("🧠 Setting up conversation memory...")
        self.memory = MemoryManager()
        
        # Initialize tools
        print("🔧 Configuring tools...")
        web_search = WebSearchTool(max_results=5)
        calculator = CalculatorTool()
        
        self.tools = [
            web_search.as_tool(),
            calculator.as_tool()
        ]
        
        # Initialize agent
        print("🤖 Creating research agent...")
        self.agent = ResearchAgent(
            tools=self.tools,
            vector_store_manager=self.vector_store,
            memory_manager=self.memory
        )
        
        print("✅ Research Assistant ready!\n")
    
    def load_documents(self):
        """Load documents from the data directory."""
        docs_path = config.DOCS_DIR
        
        if not any(docs_path.iterdir()):
            print(f"⚠️  No documents found in {docs_path}")
            print(f"   Place your PDF or TXT files in this directory to load them.")
            return
        
        print(f"📖 Loading documents from {docs_path}...")
        count = self.vector_store.load_documents()
        
        if count > 0:
            print(f"✓ Successfully loaded {count} document chunks")
        else:
            print("⚠️  No documents were loaded")
    
    def query_documents(self, question: str):
        """Query the document knowledge base."""
        print(f"\n🔍 Searching documents for: '{question}'")
        result = self.agent.query_documents(question)
        
        print(f"\n📄 Answer: {result['answer']}")
        
        if result['sources']:
            print("\n📚 Sources:")
            for idx, source in enumerate(result['sources'], 1):
                print(f"   {idx}. {source.get('source', 'Unknown')}")
    
    def research(self, query: str):
        """Perform comprehensive research."""
        print(f"\n🔬 Researching: '{query}'")
        print("=" * 60)
        
        response = self.agent.research(query)
        
        print("=" * 60)
        print(f"\n💡 Final Answer:\n{response}\n")
    
    def interactive_mode(self):
        """Run in interactive chat mode."""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE - Type 'quit' to exit")
        print("Commands:")
        print("  /docs [query]  - Search your document knowledge base")
        print("  /load          - Reload documents from data directory")
        print("  /clear         - Clear conversation history")
        print("  /help          - Show this help message")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.startswith('/docs'):
                    query = user_input[6:].strip()
                    if query:
                        self.query_documents(query)
                    else:
                        print("⚠️  Please provide a query after /docs")
                    continue
                
                if user_input == '/load':
                    self.load_documents()
                    continue
                
                if user_input == '/clear':
                    self.memory.clear()
                    print("✓ Conversation history cleared")
                    continue
                
                if user_input == '/help':
                    print("\nCommands:")
                    print("  /docs [query]  - Search your document knowledge base")
                    print("  /load          - Reload documents from data directory")
                    print("  /clear         - Clear conversation history")
                    print("  /help          - Show this help message\n")
                    continue
                
                # Regular research query
                self.research(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main entry point."""
    assistant = ResearchAssistant()
    
    # Load any documents in the data directory
    assistant.load_documents()
    
    # Enter interactive mode
    assistant.interactive_mode()

if __name__ == "__main__":
    main()
