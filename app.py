import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the API keys from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Set LangChain API key for use in API calls
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Streamlit app title and description
st.set_page_config(page_title="AI-Powered Document Summarizer", page_icon="📄", layout="centered")
st.title("📄 AI-Powered Document Summarizer")
st.markdown("""
    Welcome to the **AI-Powered Document Summarizer**! This tool uses advanced AI to summarize your PDF documents quickly and accurately. 
    Upload a PDF (max 20 pages), and let the AI do the rest.
""")

# Divider for visual separation
st.divider()

# File uploader for user to upload their PDF document (max 20 pages)
uploaded_file = st.file_uploader("**Upload a PDF document (max 20 pages)**", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF file temporarily
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the PDF document using PyPDFLoader
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Check if the document has more than 20 pages
        if len(documents) > 20:
            st.error("❌ The document exceeds the 20-page limit. Please upload a smaller document.")
        else:
            # Define text splitter to split document into chunks of size 500 with overlap of 100 characters
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

            # Splitting the documents into chunks
            chunks = text_splitter.split_documents(documents)
            st.success(f"✅ Document processed successfully! Total chunks created: **{len(chunks)}**")

            # Initialize Ollama Embeddings for embedding model
            embedding_model = OllamaEmbeddings(model="nomic-embed-text")

            # Create a FAISS vector store from the document chunks
            vector_store = FAISS.from_documents(chunks, embedding_model)

            # Save the FAISS index locally
            vector_store.save_local("faiss_index")

            # Load the FAISS index from the local storage (with dangerous deserialization allowed)
            vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

            # Initialize Groq's LLaMA model
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

            # Create a retriever from the FAISS vector store
            retriever = vector_store.as_retriever()

            # Define a custom prompt for summarization
            summarization_prompt = PromptTemplate(
                input_variables=["context"],
                template="""
                You are an expert document summarizer. Summarize the following document content in a concise and informative way:
                
                {context}
                
                Summary:
                """
            )

            # Define the summarization chain with the retriever and Groq's LLaMA model
            summarization_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": summarization_prompt},
                return_source_documents=True
            )

            # Button to trigger summarization
            if st.button("**Summarize Document**", type="primary"):
                with st.spinner("🔍 Analyzing the document and generating summary..."):
                    # Perform the summarization
                    result = summarization_chain.invoke({"query": "Summarize the document."})

                    # Display the summary
                    st.subheader("📝 Document Summary")
                    st.success(result["result"])

                    # Show the sources from where the summary was generated (first 200 characters of each)
                    st.subheader("📚 Source Documents")
                    for i, doc in enumerate(result["source_documents"], start=1):
                        st.markdown(f"**Source {i}:** {doc.page_content[:200]}...")
                        st.divider()

    except Exception as e:
        st.error(f"❌ An error occurred while processing the PDF: {e}")

else:
    st.info("ℹ️ Please upload a PDF document (max 20 pages) to begin the summarization.")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center;">
        <p>Built with ❤️ using <a href="https://streamlit.io/">Streamlit</a>, <a href="https://python.langchain.com/">LangChain</a>, and <a href="https://groq.com/">Groq</a></p>
    </div>
""", unsafe_allow_html=True)