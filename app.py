import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader, CSVLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, CTransformers
from langchain.prompts import PromptTemplate
import tempfile
from dotenv import load_dotenv  # Import dotenv to load the .env file

# Load environment variables from the .env file
# load_dotenv()
load_dotenv("myconfig.env") 

# Access your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
print(f"OpenAI API Key: {openai_api_key}")

# Function to load documents based on file type
def load_documents(root_dir, file_type):
    documents = []
    
    if file_type == "url":
        # Load URLs from a text file
        url_file_path = os.path.join(root_dir, "urls.txt")
        if not os.path.exists(url_file_path):
            raise FileNotFoundError(f"URL file not found at {url_file_path}.")
        
        with open(url_file_path, "r") as file:
            urls = [line.strip() for line in file.readlines() if line.strip()]
        
        if not urls:
            raise ValueError("No URLs found in the URL file.")
        
        # Load documents from URLs
        loader = UnstructuredURLLoader(urls=urls)
        documents.extend(loader.load())
    
    else:
        # Traverse the root directory to find all subdirectories
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            
            if os.path.isdir(subdir_path):
                # Load all files in the subdirectory
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    
                    if file_type == "pdf" and file_name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif file_type == "csv" and file_name.endswith(".csv"):
                        loader = CSVLoader(file_path=file_path)
                    elif file_type == "word" and file_name.endswith(".docx"):
                        loader = UnstructuredFileLoader(file_path)
                    elif file_type == "txt" and file_name.endswith(".txt"):
                        loader = UnstructuredFileLoader(file_path)
                    else:
                        continue  # Skip unsupported file types
                    
                    # Load documents from the file
                    documents.extend(loader.load())
    
    if not documents:
        raise ValueError(f"No documents found for file type: {file_type}.")
    
    return documents

# Function to initialize LLM
def initialize_llm(llm_type, openai_api_key):
    if llm_type == "OpenAI":
        if openai_api_key is None:
            st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
            return None
        return OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    elif llm_type == "LLaMA 2":
        return CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin', model_type='llama')
    elif llm_type == "Mistral":
        return CTransformers(model='models/mistral-7b-instruct-v0.1.Q4_K_M.gguf', model_type='mistral')
    else:
        raise ValueError("Unsupported LLM type.")

# Streamlit App
def main():
    st.set_page_config(page_title="DataQuestAI", page_icon="❄️", layout="centered")
    st.title("DataQuestAI ❄️")

    # Sidebar for configuration
    st.sidebar.title("Configuration")
    file_type = st.sidebar.selectbox("Select Document Type", ["pdf", "url", "csv", "word", "txt"])
    llm_type = st.sidebar.selectbox("Select LLM", ["OpenAI", "LLaMA 2", "Mistral"])

    # File upload or directory input
    root_dir = st.sidebar.text_input("Enter the root directory path:", "Enter your data directory path")
    if not root_dir:
        st.sidebar.warning("Please provide the root directory path.")
    else:
        if not os.path.exists(root_dir):
            st.sidebar.error("The specified root directory does not exist.")

    # Fetch the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Load the key from the .env file

    # Process documents
    if root_dir and st.sidebar.button("Process Documents"):
        try:
            # Load documents
            documents = load_documents(root_dir, file_type)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # Generate embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) if llm_type == "OpenAI" else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Save vectorstore for future use
            vectorstore.save_local("vectorstore")
            
            st.sidebar.success("Documents processed and vectorstore created!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    # Query input
    query = st.text_input("Enter your question:")
    if query:
        if os.path.exists("vectorstore"):
            try:
                # Load vectorstore
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) if llm_type == "OpenAI" else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local("vectorstore", embeddings)
                
                # Initialize LLM
                llm = initialize_llm(llm_type, openai_api_key)
                if llm is None:
                    return  # Stop execution if LLM initialization fails
                
                # Create RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                # Generate response
                response = qa_chain({"query": query})
                st.write("Answer:", response["result"])
                st.write("Sources:")
                for doc in response["source_documents"]:
                    st.write(doc.metadata["source"])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please process documents first.")

# Run Streamlit app
if __name__ == "__main__":
    main()
