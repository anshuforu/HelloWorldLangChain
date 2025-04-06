📚 Knowledge Extractor Agent (PDF + LLM + RAG)
An intelligent PDF-powered chatbot that can answer questions from one or more documents using LangChain, ChromaDB, and Hugging Face LLMs. Ideal for building private, local knowledge bases for internal docs, product manuals, SOPs, or academic papers.

How It Works

    Add your PDF file names to pdf_config.txt

    On first run:

        PDFs are split into chunks

        Embeddings are generated

        Data is stored in a persistent Chroma vector DB

    Ask any question about the documents

    The agent retrieves relevant chunks and sends them to an LLM

    You get an answer + source context!
