import hashlib
import os
import lancedb
import pymupdf4llm
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


# 1. Global Database Setup

DB_PATH = os.path.expanduser("~/.student_rag/lancedb")
os.makedirs(DB_PATH, exist_ok=True)


# 2. Universal File-to-Markdown Parser

def extract_markdown(file_path: str) -> str:
    """
    Identifies the file type and converts content into clean Markdown.
    """

    extension = os.path.splitext(file_path)[1].lower()
   
    if extension == '.pdf':
        return pymupdf4llm.to_markdown(file_path)

    elif extension in ['.docx','.pptx','.txt','.md']:
        mid = MarkItDown()

        result = mid.convert(file_path)
        return (result.text_content)


    else :
        raise ValueError(f"Unsupported file extetion {extension}")


# 3. File Hash Computation (For Deduplication)

def compute_sha256(file_path: str) -> str:
    
    # Generates a unique SHA-256 signature for a file so we don't process the same file twice.
   
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
    

def ingest_file(file_path: str, chat_id: str) -> dict:
    """
    Orchestrates parsing, hashing, chunking, embedding, and storage.
    """
    # 1. Get the hash and the filename
    file_hash = compute_sha256(file_path)
    filename = os.path.basename(file_path)
    
    # 2. Database Connection
    db = lancedb.connect(DB_PATH)
    
    # 3. Check for duplicates
    table = None

    try:
        table = db.open_table('documents')
        exsist = table.search().where(f"file_hash = '{file_hash}'").limit(1).to_list()

        if exsist:
            return{
                'status': "already_indexed",
                "filename":filename,
                "file_hash":file_hash
                }
    except Exception as e:

        print(f"\n--- DEBUG: FAILED TO OPEN TABLE ---")
        print(f"The error is: {e}\n")

        pass
    # 4. Extract Markdown

    markdown_cont = extract_markdown(file_path)

    if not markdown_cont or not markdown_cont.strip():

        return {
            "status": "empty_file",
            "filename": filename
        }
    # 5. Chunk the Text
    rctp = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    

    #Use the splitter to split the markdown text into a list of strings (chunks).
    chunks = rctp.split_text(markdown_cont)
    
    # 6. Embeddings (Vectorization)
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    vectors =  embedding_model.embed_documents(chunks)


    records = []

    for idx, (chuck_text,vector) in enumerate(zip(chunks,vectors)):

        records.append({
            "id": f"{file_hash}_{idx}",
            "vector": vector,
            "text": chuck_text,
            "source": filename,
            "chat_id": chat_id,
            "file_hash": file_hash
        })
    # 8. Save to Database
    
    if table is None:
        db.create_table('documents',data=records)
    else:
        table.add(records)

    return {
        "status": "success",
        "chunks_indexed": len(records), 
        "filename": filename
    }

if __name__ == "__main__":
    result = ingest_file(r"pdf docs for rag\PS notes.pdf","chat_001")
    print(result)