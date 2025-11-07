
import logging
import os
import sys
import asyncio
from pathlib import Path
from io import BytesIO

# Fix pour Windows: utiliser SelectorEventLoop au lieu de ProactorEventLoop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables FIRST, before any imports that depend on them
from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import json
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tavily import TavilyClient
import psycopg2
from openai import OpenAI
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

from crag_graph import get_crag_graph

# Configuration PostgreSQL pour PGVector
postgres_connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

app = FastAPI(title="Dagan Agent RAG API", version="2.0.0")

# Middleware CORS complet
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "everything is ok"}

class VectorizeRequest(BaseModel):
    url: str

class CragQueryRequest(BaseModel):
    question: str
    conversation_id: str = None

@app.get("/")
async def ping():
    return {"message": "Alive"}

# ============================================================================
# ENDPOINTS DE VECTORISATION (UPLOAD)
# ============================================================================

@app.post("/api/upload")
async def vectorize_pdf_file(
    file: UploadFile = File(...),
    collection_name: str = Form("pdf_uploads")
):
    """
    Vectorise le contenu d'un fichier PDF en chunks avec embeddings.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Format de fichier non supporté. Uniquement .pdf accepté."
            )
        
        logging.info(f"Début de la vectorisation pour le fichier : {file.filename}")

        file_content = await file.read()
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / file.filename

        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        loader = PyPDFLoader(str(temp_file_path))
        documents_from_pdf = loader.load()
        text_content = "\n".join([doc.page_content for doc in documents_from_pdf])
        
        os.remove(temp_file_path)

        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Le PDF est vide ou ne contient pas de texte extractible."
            )
        
        file_size = len(file_content)
        logging.info(f"Fichier '{file.filename}' lu avec succès ({file_size} bytes, {len(text_content)} caractères)")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Aucun chunk généré. Le contenu est peut-être trop court."
            )
        logging.info(f"{len(chunks)} chunks générés.")
        
        collection = collection_name
        upload_timestamp = datetime.utcnow().isoformat()
        
        documents_to_embed = []
        for chunk_index, chunk_content in enumerate(chunks):
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_date": upload_timestamp,
                    "file_type": "application/pdf",
                    "source": "pdf_upload",
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks)
                }
            )
            documents_to_embed.append(doc)
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding VECTOR(2000),
                document TEXT,
                cmetadata JSONB
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
            ON langchain_pg_embedding 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        conn.commit()
        
        uuids = [str(uuid4()) for _ in range(len(documents_to_embed))]
        
        for doc, doc_id in zip(documents_to_embed, uuids):
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=doc.page_content,
                dimensions=2000
            )
            embedding = response.data[0].embedding
            
            cursor.execute("""
                INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    document = EXCLUDED.document, 
                    cmetadata = EXCLUDED.cmetadata
            """, (doc_id, collection, embedding, doc.page_content, json.dumps(doc.metadata)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"{len(documents_to_embed)} documents vectorisés et stockés dans la collection '{collection}'")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Fichier '{file.filename}' vectorisé avec succès.",
                "filename": file.filename,
                "collection": collection,
                "documents_count": len(documents_to_embed)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vectorisation du fichier PDF: {str(e)}"
        )

@app.post("/vectorize-file")
async def vectorize_file(
    file: UploadFile = File(...),
    collection_name: str = Form(None)
):
    """
    Vectorise le contenu d'un fichier texte (.txt) en chunks avec embeddings.
    """
    try:
        if not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=400,
                detail="Format de fichier non supporté. Uniquement .txt accepté."
            )
        
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024: # 10 MB
            raise HTTPException(
                status_code=400,
                detail="Fichier trop volumineux. Maximum: 10 MB."
            )
        
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('latin-1')
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Le fichier est vide ou ne contient pas de texte valide."
            )
        
        logging.info(f"Fichier '{file.filename}' lu avec succès.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Aucun chunk généré."
            )
        
        logging.info(f"{len(chunks)} chunks générés.")
        
        collection = collection_name or "file_uploads"
        upload_timestamp = datetime.utcnow().isoformat()
        
        documents = []
        for chunk_index, chunk_content in enumerate(chunks):
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_date": upload_timestamp,
                    "file_type": "text/plain",
                    "source": "file_upload",
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks)
                }
            )
            documents.append(doc)
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding VECTOR(2000),
                document TEXT,
                cmetadata JSONB
            );
        """)
        conn.commit()
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        for doc, doc_id in zip(documents, uuids):
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=doc.page_content,
                dimensions=2000
            )
            embedding = response.data[0].embedding
            
            cursor.execute("""
                INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    document = EXCLUDED.document, 
                    cmetadata = EXCLUDED.cmetadata
            """, (doc_id, collection, embedding, doc.page_content, json.dumps(doc.metadata)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"{len(documents)} documents vectorisés dans la collection '{collection}'")
        
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "collection": collection,
                "documents_count": len(documents)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize")
async def vectorize_url(body: VectorizeRequest):
    """
    Vectorize a URL by crawling it and creating embeddings.
    """
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        crawl_result = tavily_client.crawl(url=body.url, format="text", limit=4)
        combined_content = "".join([res.get("raw_content", "") for res in crawl_result["results"]])

        if not combined_content.strip():
            raise HTTPException(status_code=400, detail="No content found to vectorize")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,           
            chunk_overlap=800,         
            separators=["\n\n", "\n", ". ", " ", ""], 
            length_function=len
        )
        chunks = text_splitter.split_text(combined_content)

        documents = [Document(page_content=chunk) for chunk in chunks]
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        collection_name = "crawled_documents"

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding VECTOR(2000),
                document TEXT,
                cmetadata JSONB
            );
        """)
        conn.commit()
        
        uuids = [str(uuid4()) for _ in range(len(documents))]

        for doc, doc_id in zip(documents, uuids):
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=doc.page_content,
                dimensions=2000
            )
            embedding = response.data[0].embedding
            
            cursor.execute("""
                INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                VALUES (%s, %s, %s::vector, %s, %s)
            """, (doc_id, collection_name, embedding, doc.page_content, json.dumps({"url": body.url})))
        
        conn.commit()
        cursor.close()
        conn.close()

        return JSONResponse(content={"success": True, "documents_count": len(documents)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS CRAG/RAG
# ============================================================================

@app.post("/crag/query")
async def crag_query(body: CragQueryRequest):
    try:
        thread_id = body.conversation_id or str(uuid4())
        agent_graph = get_crag_graph()
        
        initial_state = {
            "messages": [HumanMessage(content=body.question)],
            "question": body.question,
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        final_state = agent_graph.invoke(initial_state, config)
        
        messages = final_state.get("messages", [])
        final_answer = messages[-1].content if messages and hasattr(messages[-1], 'content') else "No answer generated."
        sources = final_state.get("sources", [])
        
        return JSONResponse(content={
            "success": True,
            "conversation_id": thread_id,
            "answer": final_answer,
            "sources": sources
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crag/stream")
async def crag_stream(body: CragQueryRequest):
    
    async def event_generator():
        try:
            thread_id = body.conversation_id or str(uuid4())
            agent_graph = get_crag_graph()
            config = {"configurable": {"thread_id": thread_id}}

            state_snapshot = agent_graph.get_state(config)
            old_messages = state_snapshot.values.get("messages", []) if state_snapshot else []
            
            initial_state = {"messages": old_messages + [HumanMessage(content=body.question)], "question": body.question}
            
            accumulated_answer = ""
            collected_sources = []

            async for event in agent_graph.astream(initial_state, config):
                for node_name, node_output in event.items():
                    if "messages" in node_output:
                        current_messages = node_output.get("messages", [])
                        if current_messages:
                            last_message = current_messages[-1]
                            if hasattr(last_message, 'content'):
                                chunk = last_message.content
                                accumulated_answer += chunk
                                yield json.dumps({"type": "message_chunk", "content": chunk}) + "\n"
            
            # Final event with all data
            yield json.dumps({
                "type": "complete",
                "conversation_id": thread_id,
                "answer": accumulated_answer,
                "sources": collected_sources # This needs to be populated correctly from the graph state
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
