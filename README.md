# RAG-Retrieval-Augmented-Generation
This repo uses LLMs to summarize documents (PDF, URL, CSV, Word, TXT) and answer questions based on the summaries. It loads files, splits text, creates embeddings, and builds a retrieval-based QA system with OpenAI, LLaMA 2, or Mistral models, all accessible via a Streamlit app.
Key features include:

1. Loading and processing diverse document types from local directories or URLs.

2. Splitting documents into manageable chunks for efficient embedding and retrieval.

3. Creating vector embeddings using OpenAI or HuggingFace models for semantic search.

4. Building a retrieval-based QA system powered by LLMs such as OpenAI GPT-3.5 Turbo, LLaMA 2, or Mistral models.

5. An interactive Streamlit interface for document processing, querying, and displaying answers with source references.

The system leverages LangChain for document loading, chunking, embedding, and retrieval chaining, providing a scalable and modular approach to building knowledge-driven AI applications.

- Data root directory:
<img src="https://github.com/krmanu07/RAG-Retrieval-Augmented-Generation-/blob/main/Output/data%20directory.png?raw=true" width="450" height="250">

- Home Page
<img src="https://github.com/krmanu07/RAG-Retrieval-Augmented-Generation-/blob/main/Output/HomePage.png?raw=true" width="850" height="450">

- Select your file type you want to process your data for Question-Answering
<img src="https://github.com/krmanu07/RAG-Retrieval-Augmented-Generation-/blob/main/Output/FileType.png?raw=true" width="850" height="450">

- Select your LLMs type
<img src="https://github.com/krmanu07/RAG-Retrieval-Augmented-Generation-/blob/main/Output/LLMType.png?raw=true" width="850" height="450">
