from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------ Config ------------------
PDF_FOLDER = "./PDF"
VECTOR_DB_FOLDER = "./vector_dbs"   # Local storage for embeddings
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
llm = ChatOllama(model="llama3.2")
#llama3.2 mistral
# Store vector DBs by filename in memory (but backed by disk)
vector_dbs = {}

app = FastAPI()

# ------------------ Utils ------------------
def safe_collection_name(file_name: str) -> str:
    base = os.path.splitext(file_name)[0]
    safe = re.sub(r'[^a-zA-Z0-9._-]', "_", base)
    safe = safe.strip("._-")
    if len(safe) < 3:
        safe = f"col_{safe}"
    return safe

def process_pdf(file_path: str, file_name: str):
    """Load, split, and persist PDF into its own Chroma vectorstore."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    collection_name = safe_collection_name(file_name)
    persist_path = os.path.join(VECTOR_DB_FOLDER, collection_name)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_path  # <-- saves to disk
    )
    vector_db.persist()  # <-- ensures embeddings are stored
    vector_dbs[file_name] = vector_db

def load_existing_vector_dbs():
    """Reload vector DBs from disk on server restart."""
    for folder in os.listdir(VECTOR_DB_FOLDER):
        persist_path = os.path.join(VECTOR_DB_FOLDER, folder)
        if os.path.isdir(persist_path):
            db = Chroma(
                embedding_function=embeddings,
                collection_name=folder,
                persist_directory=persist_path,
            )
            vector_dbs[folder + ".pdf"] = db  # keep consistent with file name

# ------------------ Startup ------------------
@app.on_event("startup")
async def startup_event():
    load_existing_vector_dbs()

# ------------------ API ------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a single PDF, store embeddings locally."""
    file_path = os.path.join(PDF_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    process_pdf(file_path, file.filename)

    return {"message": f"{file.filename} uploaded & indexed successfully."}

@app.post("/askWith/")
async def ask_questionWith(payload: dict = Body(...)):
    question = payload.get("question")
    if not question:
        return {"error": "Missing question"}

    if not vector_dbs:
        return JSONResponse(
            status_code=400,
            content={"error": "No PDFs uploaded yet."}
        )

    retrievers = [db.as_retriever() for db in vector_dbs.values()]
    all_contexts = []
    for retriever in retrievers:
        docs = retriever.get_relevant_documents(question)
        all_contexts.extend(docs)

    context_text = "\n\n".join([doc.page_content for doc in all_contexts])

    template = """You are a helpful assistant. 
    Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": lambda _: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"question": question})
    return {"answer": answer}

@app.get("/list_pdfs/")
async def list_pdfs():
    """List uploaded PDFs (vector DBs loaded from disk)."""
    return {"pdfs": list(vector_dbs.keys())}
