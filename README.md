---

# GROQ Chat with Llama 3 & Document Retrieval

This is a Streamlit application that utilizes **GROQ's Llama 3** model to answer user questions based on uploaded PDF documents. It uses **Langchain** for document loading and vector embeddings, and **FAISS** for similarity search to provide accurate responses.

## Features

- **PDF Document Loading**: Load PDF documents from a specified directory.
- **Text Splitting**: Split documents into smaller chunks for more efficient embeddings and retrieval.
- **Embedding Creation**: Generate embeddings using **OpenAI** to vectorize documents.
- **Retrieval-Based Question Answering**: Use a **retrieval chain** to find relevant documents and answer questions based on them.
- **Chat with Llama 3**: Interact with Groq's **Llama 3** model, which retrieves context from relevant documents and generates responses.

## Prerequisites

Before running the application, ensure you have the following:

- Python 3.8 or higher
- [Streamlit](https://docs.streamlit.io/)
- [Langchain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Groq](https://www.groq.com/)
- OpenAI and Groq API Keys

### Python Packages

Install the required packages via pip:

```bash
pip install -r requirements.txt
```

## Setting Up API Keys

1. Create a `.env` file in the root directory of your project with the following content:

```
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
```

2. Alternatively, you can add your API keys to the `secrets.toml` file used by Streamlit. For example:

```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
GROQ_API_KEY = "your-groq-api-key"
```

## Folder Structure

Make sure you have your PDF documents in the correct folder:

```
/your_project
    /pages
    /all_docs
        document1.pdf
        document2.pdf
    main.py
    README.md
    .env
```

## Usage

1. **Initialize the Vector Store:**

   Click the "initialize vector store" button to load documents and create embeddings. The system splits the documents into manageable chunks for processing.

2. **Ask Questions:**

   Enter a question related to the loaded documents into the text input box, and click submit. The app will return the most relevant response from the documents.

3. **View Similarity Search Results:**

   If you want to inspect the chunks retrieved from the document, expand the "Show Similarity Search" section, where you'll see the matching document chunks.

## How It Works

1. **Document Loading**: Uses `PyPDFDirectoryLoader` to load documents from a specified directory.
2. **Text Splitting**: Splits documents into chunks (with some overlap) for more efficient embedding generation.
3. **Embedding Creation**: Generates embeddings using OpenAI's model via the `OpenAIEmbeddings` class.
4. **FAISS Index**: Stores document embeddings in a FAISS vector store for fast retrieval.
5. **Question Answering**: Uses Langchain's `ChatGroq` to answer user questions based on relevant document chunks found via FAISS similarity search.

## Performance

- The system provides the response time for each question asked, helping you gauge the speed of the retrieval and response generation.
  
## Running the App

To start the app:

```bash
streamlit run main.py
```

This will launch the Streamlit app in your web browser.

## Contributing

Feel free to open a pull request if you have any suggestions or improvements!

---

### License

This project is licensed under the MIT License.

---

This `README.md` gives a comprehensive overview of your Streamlit app and how to use it. Let me know if you want to add or adjust any details!