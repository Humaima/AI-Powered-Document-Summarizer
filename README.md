# AI-Powered-Document-Summarizer
This is an AI-powered document summarizer built using Streamlit, LangChain, and Groq. It allows users to upload a PDF document (up to 20 pages) and generates a concise summary using advanced AI models.
## Features
- Upload and process PDF documents.
- Summarize documents using Groq's LLaMA model.
- Display the summary and source chunks.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-powered-document-summarizer.git
   cd ai-powered-document-summarizer

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Set up environment variables:**
   
   Create a .env file in the root directory.
   
   Add your API keys:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
6. **Access the app:**
   
   Open your browser and navigate to http://localhost:8501.
   
