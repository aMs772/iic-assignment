# IIC Projects and Initiatives Division Coordinator Recruitment Assignment

### I have choosen option 1 (technical build).
In this option I have chose to build a RAG chatbot.

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **ChromaDB**, and **OpenAI**. The chatbot retrieves relevant information from a custom knowledge base and uses an LLM to generate grounded responses based only on the retrieved context.

A simple website is also coded to display the prototype in GUI mode, instead of cli mode.


## Tech Stack

- **Language:** Python 3.14
- **LLM:** OpenAI GPT-4o Mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Document Processing:** RecursiveCharacterTextSplitter
- **Configuration:** python-dotenv
- **Package Manager:** uv
- **local website(GUI Mode):** gradio

---

## Project Structure

```text
.
├── .venv/                  # Virtual environment
├── __pycache__/
│
├── .env                    # OpenAI API key
├── .gitignore
├── .python-version
│
├── app.py                  # Application entry point
├── main.py                 # Alternate entry point
├── rag.py                  # Core RAG implementation
│
├── knowledge.txt           # IIC Coordinator recruitment knowledge base written by chat GPT.
├── final_knowledge.txt     # IIC Coordinator recruitment slides and pdf extracted in text format.
├── log.txt                 # Retrieved context logs
├── harry_potter.txt        # knowledge base of harry potter world written by claude
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## RAG Pipeline

```
Knowledge Base
      │
      ▼
Load Document
      │
      ▼
Chunk Document
      │
      ▼
Generate Embeddings
      │
      ▼
Store in ChromaDB
      │
      ▼
User Query
      │
      ▼
Similarity Search
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Prompt Construction
      │
      ▼
GPT-4o Mini
      │
      ▼
Final Response
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment.

```bash
uv venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
uv sync
```

---

## Environment Variables

### NOTE:
I have used OpenAI api key, for getting an openAI api key, go to this website and get a private key. 
But Unfortunately for getting an OpenAI key, You need to buy minimum 5$ credits 😞.

```
https://openai.com/index/openai-api/
```
create a `.env` file and paste your OpenAI API key as shown below.

```env
OPENAI_API_KEY=your_openai_api_key
```
But if you want to use a free api key for testing this purpose, then you can use a free google api key.
For getting a free google api key go to this website (google ai studio) and in api keys section you can create a new api key for free.

```
https://aistudio.google.com/api-keys
```
then create a ``.env`` file and paste your GOOGLE API KEY as 
```env
GOOGLE_API_KEY=your google api key
```

---

## Adding your custom knowledge base

currently the rag.py supports only a single text document.
So if you want to use multiple text files, first convert them into a single text file and add that file to the directory

## Running the Chatbot

### 1. CLI mode:
If you want to use the chat bot in cli mode, run the following command

```bash
python main.py <your knowledge base file name>
```
You will enter a loop where you can type the query and get response.
To exit the loop type `exit`

### 2. GUI mode:
IF you want to use the chat bot in GUI mode, first edit the app.py file
In line number 9, add your custom knowledge base file path( or just name if it is in the current directory itself)

```
FILE_PATH = "<filename.txt>"
```

then run the following command
```bash
python app.py
```
You will get an url, copy that link and paste it in your favourite browser, and from there you can acsess the chatbot

---

## How It Works

1. The knowledge base is loaded.
2. The document is split into overlapping chunks.
3. Each chunk is converted into an embedding.
4. Chunks are stored in a Chroma vector database.
5. User queries are embedded.
6. The most relevant chunks are retrieved.
7. Retrieved context is inserted into the prompt.
8. GPT-4o Mini generates the final answer using only the retrieved context.

---

## Retrieval Logging

Every query stores:

* User query
* Retrieved document chunks

These are written to:

```text
log.txt
```

This is useful for debugging retrieval quality.

---

## Current Knowledge Base

The chatbot currently uses a custom text knowledge base.

I have initially thought to make a helper bot which helps students aiming to become coordinators of IIC, 
but I didnt have proper formatted data, so the RAG results were not upto the mark.
Hence I switched to a bigger and cleaner document of harry potter universe (written by CLAUDE).

currently the project can handle only a single text based knowledge document but the project can easily be adapted to support:

* PDF documents
* Multiple text files
* Markdown documents
* Web pages
* Custom datasets

---

## Future Improvements

* Persistent Chroma database
* PDF document support
* Multi-document retrieval
* Hybrid search (keyword + vector search)
* Chat history
* Modern web interface

---

# THANK YOU 

---
