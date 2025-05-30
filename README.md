# RAG-Retrieval-Augmented-Generation
his repo uses LLMs to summarize documents (PDF, URL, CSV, Word, TXT) and answer questions based on the summaries. It loads files, splits text, creates embeddings, and builds a retrieval-based QA system with OpenAI, LLaMA 2, or Mistral models, all accessible via a Streamlit app.
Key features include:

Loading and processing diverse document types from local directories or URLs.

Splitting documents into manageable chunks for efficient embedding and retrieval.

Creating vector embeddings using OpenAI or HuggingFace models for semantic search.

Building a retrieval-based QA system powered by LLMs such as OpenAI GPT-3.5 Turbo, LLaMA 2, or Mistral models.

An interactive Streamlit interface for document processing, querying, and displaying answers with source references.

The system leverages LangChain for document loading, chunking, embedding, and retrieval chaining, providing a scalable and modular approach to building knowledge-driven AI applications.
