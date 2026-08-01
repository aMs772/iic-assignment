# import external libraries
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# comment line 7 and uncomment line 8 if you want to use OpenAI instead of Google Generative AI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# import standard libraries
import sys

print("imported all libraries successfully")

# initialize 
load_dotenv()

# uncomment lines 23, 24 and uncomment line 26, 27 if you want to use OpenAI instead of Google Generative AI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 150,
)

def get_documment(file_path: str) -> Document:
    """
    Get a document from a file path.
    Args:file_path (str): The path to the file.
    Returns:Document: The document object.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Document(
        page_content=content, 
        metadata={
            "source": file_path
            }
        )
    return doc

def get_chunks(doc: Document) -> list[Document]:
    """
    Get chunks from a document.
    Args:doc (Document): The document object.
    Returns:list[Document]: A list of document chunks.
    """

    chunks = splitter.split_documents([doc])
    return chunks

def create_vector_store(chunks: list[Document]):
    """
    Create a vector store from document chunks.
    Args:chunks (list[Document]): A list of document chunks.
    Returns:VectorStore: The vector store object.
    """

    # from langchain_vectorstores import FAISS
    # vector_store = FAISS.from_documents(chunks, llm)
    # return vector_store

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings
    )
    
    return vector_store

def rag_init(file_path: str):
    """
    Initialize the RAG pipeline.
    Args:file_path (str): The path to the file.
    Returns:VectorStore: The vector store object.
    """
    try:
        doc = get_documment(file_path)
    except Exception as e:
        print(f"Error reading document: {e}")
        sys.exit(1)
    chunks = get_chunks(doc)
    vector_store = create_vector_store(chunks)
    
    return vector_store

def rag_query(vector_store, query: str):
    """
    Query the RAG pipeline.
    Args:vector_store (VectorStore): The vector store object.
         query (str): The query string.
    Returns:str: The response from the RAG pipeline.
    """

    # from langchain_vectorstores import FAISS
    # docs = vector_store.similarity_search(query)

    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4}
    )

    def log_query(query: str) -> str:
        """
        Log the query to a file.
        Args:query (str): The query string.
        Returns:str: The query string.
        """
        with open("log.txt", "a") as f:
            f.write(f"Query: {query}\n\n")
        return query

    def log_docs_and_format(docs: list[Document]) -> str:
        """
        Log the documents to a file.
        Args:docs (list[Document]): A list of document chunks.
        Returns:str: A formatted string of the document chunks.
        """
        with open("log.txt", "a") as f:
            for doc in docs:
                f.write(f"Document: {doc.page_content}\n\n")
        
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
        
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the following context:
{context}

Question: {query}

and if the answer is not contained within the context, say "I don't know."

"""
    )

    rag_chain = (
        {
            "context": RunnableLambda(log_query) | retriever | RunnableLambda(log_docs_and_format),
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    return answer

def main():
    """
    Main function to run the RAG pipeline.
    """
    if len(sys.argv) < 2:
        print("Usage: python rag.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    vector_store = rag_init(file_path)

    print("RAG pipeline initialized. You are now in a loop where you can query the RAG pipeline. Type 'exit' to quit.")

    while True:
        query = input("Query: ")
        if query.lower() == 'exit':
            break
        answer = rag_query(vector_store, query)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
