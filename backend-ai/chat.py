import os
import lancedb
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import LanceDB
from langchain_lancedb import LanceDB
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

DB_PATH = os.path.expanduser("~/.student_rag/lancedb")

def initialize_components():
    """
    Initializes and returns the three core pieces of our RAG system:
    1. The Database Table (LanceDB)
    2. The Embedding Model (nomic-embed-text)
    3. The Chat LLM (qwen2.5:3b)
    """
    

    db = lancedb.connect(DB_PATH)

    
    table = db.open_table('documents')

    embedding_question = OllamaEmbeddings(model="nomic-embed-text")
    
 
    chatmodel = ChatOllama(model='qwen2.5:3b',temperature=0)
    
    
    return ( db , embedding_question , chatmodel )


def get_answer(user_query: str, chat_history: list, chat_id: str) -> str:
    db, embeddings, chatmodel = initialize_components()

    # ---------------------------------------------------------
    # STEP 1: HISTORY-AWARE QUERY REWRITING
    # If there is a chat history, ask the LLM to rewrite the question 
    # so it makes sense without the history.
    # ---------------------------------------------------------
    if chat_history:
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        # This small chain just rewrites the text
        rewriter_chain = rephrase_prompt | chatmodel | StrOutputParser()
        search_query = rewriter_chain.invoke({
            "chat_history": chat_history, 
            "input": user_query
        })
        print(f"\n--- DEBUG: Rewrote query to: '{search_query}' ---\n")
    else:
        # If no history, just use the exact question
        search_query = user_query

    # ---------------------------------------------------------
    # STEP 2: NATIVE LANCEDB SEARCH (Using the standalone query!)
    # ---------------------------------------------------------
    table = db.open_table('documents')
    query_vec = embeddings.embed_query(search_query) # Embed the REWRITTEN query
    
    raw_results = table.search(query_vec).where(f"chat_id = '{chat_id}'").limit(4).to_list()

    docs = []
    for result in raw_results:
        docs.append(Document(page_content=result["text"], metadata=result))

    # ---------------------------------------------------------
    # STEP 3: ANSWER THE QUESTION
    # ---------------------------------------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI study assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\nContext:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
    
    response = question_answer_chain.invoke({
        "context": docs,         
        "input": user_query,    # The AI still replies to the original prompt
        "chat_history": chat_history
    })

    return response


if __name__ == "__main__":

    fake_histor = []

    chat_it = "chat_001"

    user_query = "what do you know about history of pakistan"

    result = get_answer(user_query,fake_histor,chat_it)

    print("reposnse ---------")
    print(result)