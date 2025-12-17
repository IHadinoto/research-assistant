"""
Conversation memory management with context window optimization.

This file provides a lightweight fallback memory implementation if the
LangChain memory submodule is not available in the environment. The fallback
is sufficient for basic testing and interactive runs where advanced memory
features are not required.
"""
import config

try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.chat_message_histories import ChatMessageHistory

    class MemoryManager:
        """Manages conversation history with intelligent context pruning."""

        def __init__(self):
            """Initialize memory with window-based retention."""
            self.memory = ConversationBufferWindowMemory(
                k=config.MAX_MEMORY_MESSAGES,
                memory_key=config.CONVERSATION_MEMORY_KEY,
                return_messages=True,
                output_key="output",
            )

        def add_interaction(self, user_input: str, assistant_output: str):
            """Add a user-assistant interaction to memory."""
            self.memory.save_context({"input": user_input}, {"output": assistant_output})

        def get_history(self) -> str:
            return self.memory.load_memory_variables({}).get(config.CONVERSATION_MEMORY_KEY, [])

        def clear(self):
            self.memory.clear()

except Exception:
    # Fallback minimal memory manager
    class _SimpleMemory:
        def __init__(self):
            self._messages = []

        def save_context(self, inp, outp):
            self._messages.append((inp, outp))

        def load_memory_variables(self, _):
            # return a dict with the configured memory key
            return {config.CONVERSATION_MEMORY_KEY: self._messages}

        def clear(self):
            self._messages = []

    class MemoryManager:
        """Fallback memory manager used when LangChain memory isn't available."""

        def __init__(self):
            self.memory = _SimpleMemory()

        def add_interaction(self, user_input: str, assistant_output: str):
            self.memory.save_context({"input": user_input}, {"output": assistant_output})

        def get_history(self) -> str:
            return self.memory.load_memory_variables({}).get(config.CONVERSATION_MEMORY_KEY, [])

        def clear(self):
            self.memory.clear()
