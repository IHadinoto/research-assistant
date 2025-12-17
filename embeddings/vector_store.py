"""
Robust vector store manager with fallbacks.

This module tries to use the recommended packages (Ollama embeddings, Chroma,
and LangChain helpers). If those packages or specific submodules are not
available in the environment it provides minimal fallbacks so the rest of the
application can be imported and run for dry runs / tests.
"""
from typing import List, Any
import config
import os


class _FallbackCollection:
    def count(self):
        return 0


class _FallbackVectorStore:
    """A tiny in-memory stub that matches the parts of the Chroma API we use."""
    def __init__(self):
        self._collection = _FallbackCollection()

    def add_documents(self, docs: List[Any]):
        # noop fallback
        return

    def similarity_search_with_score(self, query: str, k: int = 4):
        return []

    def as_retriever(self, search_kwargs=None):
        return self

    def delete_collection(self):
        return


class VectorStoreManager:
    """
    Manages document embeddings and vector storage with persistence.

    It attempts to import and initialize a proper Chroma-backed vector store
    using Ollama embeddings. If the required libraries or submodules are not
    available, a safe fallback is used so the application can still be
    imported and run for testing or documentation purposes.
    """

    def __init__(self):
        self._use_fallback = False
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None

        # Try to import the recommended libraries. If any import fails we set
        # up fallback implementations.
        try:
            from langchain_ollama import OllamaEmbeddings
            try:
                # Newer langchain may expose text splitters in different places;
                # attempt common import paths.
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                except Exception:
                    # fallback to the newer path
                    from langchain.text_splitter import RecursiveCharacterTextSplitter

                from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
                from langchain_community.vectorstores import Chroma

                self.embeddings = OllamaEmbeddings(
                    model=config.EMBEDDING_MODEL,
                    base_url=config.OLLAMA_BASE_URL,
                )

                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                )

                # Initialize or load existing Chroma vector store
                try:
                    self.vectorstore = Chroma(
                        persist_directory=str(config.VECTOR_DIR),
                        embedding_function=self.embeddings,
                        collection_name="research_docs",
                    )
                    try:
                        count = self.vectorstore._collection.count()
                    except Exception:
                        count = 0
                    print(f"✓ Loaded existing vector store with {count} documents")
                except Exception:
                    # If initializing Chroma fails, keep fallback
                    print("⚠️  Could not initialize Chroma vector store; using fallback store")
                    self.vectorstore = _FallbackVectorStore()

                # Keep loader references for later use in load_documents
                self._PyPDFLoader = PyPDFLoader
                self._TextLoader = TextLoader
                self._DirectoryLoader = DirectoryLoader

            except Exception:
                # If submodule imports fail, fall back
                raise

        except Exception:
            # If any import fails, switch to fallback mode
            self._use_fallback = True
            print("⚠️  Missing optional dependencies for full functionality; enabling fallbacks for testing")
            self.vectorstore = _FallbackVectorStore()

    def load_documents(self, file_path: str = None) -> int:
        """
        Load documents from file or directory and add to vector store.

        Returns the number of chunks added. In fallback mode this is always 0.
        """
        if self._use_fallback:
            print("⚠️  load_documents: running in fallback mode — no documents will be indexed")
            return 0

        if file_path is None:
            file_path = str(config.DOCS_DIR)

        # Determine loader based on file type
        if file_path.endswith('.pdf'):
            loader = self._PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = self._TextLoader(file_path)
        else:
            loader = self._DirectoryLoader(
                file_path,
                glob="**/*.pdf",
                loader_cls=self._PyPDFLoader,
                show_progress=True,
            )

        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        if chunks:
            try:
                self.vectorstore.add_documents(chunks)
                print(f"✓ Added {len(chunks)} document chunks to vector store")
            except Exception:
                print("⚠️  Could not add documents to vector store; operation skipped")

        return len(chunks)

    def similarity_search(self, query: str, k: int = config.TOP_K_RESULTS) -> List:
        """Perform similarity search on vector store."""
        try:
            if self.vectorstore._collection.count() == 0:
                return []
        except Exception:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception:
            return []

    def as_retriever(self):
        """Return vectorstore as a retriever for chain integration."""
        try:
            return self.vectorstore.as_retriever(search_kwargs={"k": config.TOP_K_RESULTS})
        except Exception:
            return self.vectorstore

    def clear_store(self):
        """Clear all documents from vector store."""
        try:
            self.vectorstore.delete_collection()
            # try to reinitialize a proper store
            self.__init__()
            print("✓ Vector store cleared")
        except Exception:
            print("⚠️  Could not clear vector store in fallback mode")
