import logging
import os
import sys
import asyncio
from pathlib import Path

# Fix pour Windows: utiliser SelectorEventLoop au lieu de ProactorEventLoop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables FIRST, before any imports that depend on them
from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.ERROR, format="%(message)s")
import json
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tavily import TavilyClient
import psycopg2
import numpy as np
from openai import OpenAI
from datetime import datetime

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
    # Pas de thread_id nécessaire : documents publics partagés


class CragQueryRequest(BaseModel):
    question: str
    conversation_id: str = None  # Optional, sera généré si non fourni


@app.get("/")
async def ping():
    return {"message": "Alive"}


@app.post("/vectorize-file")
async def vectorize_file(
    file: UploadFile = File(...),
    collection_name: str = Form(None)
):
    """
    Vectorise le contenu d'un fichier texte (.txt) en chunks avec embeddings.
    
    Args:
        file: Fichier .txt à vectoriser
        collection_name: Nom de la collection (défaut: "file_uploads")
        
    Returns:
        JSON avec résumé de la vectorisation
        
    Limites:
        - Taille max: 10 MB
        - Format: .txt uniquement
        
    Process:
        1. Validation fichier (extension, taille)
        2. Lecture contenu texte
        3. Chunking avec overlap (4000 chars, 800 overlap)
        4. Génération embeddings OpenAI
        5. Stockage dans PGVector avec métadonnées
    """
    try:
        # 1. Validation de l'extension
        if not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=400,
                detail="Format de fichier non supporté. Uniquement .txt accepté."
            )
        
        # 2. Lecture du contenu
        content = await file.read()
        
        # 3. Validation de la taille (10 MB max)
        max_size = 10 * 1024 * 1024  # 10 MB
        file_size = len(content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Fichier trop volumineux ({file_size} bytes). Maximum: {max_size} bytes (10 MB)."
            )
        
        # 4. Décodage du contenu en texte
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            # Essayer avec d'autres encodages
            try:
                text_content = content.decode('latin-1')
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="Impossible de décoder le fichier. Assurez-vous qu'il s'agit d'un fichier texte valide (UTF-8 ou Latin-1)."
                )
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Le fichier est vide ou ne contient pas de texte valide."
            )
        
        print(f"✓ Fichier '{file.filename}' lu avec succès ({file_size} bytes, {len(text_content)} caractères)")
        
        # 5. Chunking avec overlap (même stratégie que /vectorize)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(text_content)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Aucun chunk généré. Le contenu est peut-être trop court."
            )
        
        print(f"✓ {len(chunks)} chunks générés avec overlap (size=4000, overlap=800)")
        
        # 6. Préparer les métadonnées
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
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk_content)
                }
            )
            documents.append(doc)
        
        print(f"✓ {len(documents)} documents créés avec métadonnées")
        
        # 7. Génération des embeddings et stockage dans PGVector
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        # Vérifier/créer la table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding VECTOR(2000),
                document TEXT,
                cmetadata JSONB
            )
        """)
        
        # Créer l'index si nécessaire
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
            ON langchain_pg_embedding 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        conn.commit()
        
        # 8. Générer embeddings et insérer
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        for i, (doc, doc_id) in enumerate(zip(documents, uuids)):
            # Générer embedding
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=doc.page_content,
                dimensions=2000
            )
            embedding = response.data[0].embedding
            
            # Insérer dans PGVector
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
        
        print(f"✓ {len(documents)} documents vectorisés et stockés dans collection '{collection}'")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Fichier '{file.filename}' vectorisé avec succès",
                "filename": file.filename,
                "file_size": file_size,
                "collection": collection,
                "documents_count": len(documents),
                "chunks_info": {
                    "chunk_size": 4000,
                    "chunk_overlap": 800,
                    "total_chunks": len(chunks),
                    "total_characters": len(text_content)
                },
                "upload_date": upload_timestamp
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vectorisation du fichier: {str(e)}"
        )

@app.post("/vectorize")
async def vectorize_url(
    body: VectorizeRequest,
):
    """
    Vectorize a URL by crawling it and creating embeddings in PostgreSQL
    with intelligent chunking (overlap between chunks for better context).
    
    Process:
    1. Crawl URL with Tavily (get raw content + favicon)
    2. Split content into chunks with overlap (RecursiveCharacterTextSplitter)
    3. Vectorize each chunk with OpenAI embeddings (direct API)
    4. Store in PGVector with metadata (url, favicon, chunk_index, chunk_count)
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # 1. Crawl URL
        crawl_result = tavily_client.crawl(
            url=body.url, format="text", include_favicon=True, limit=4
        )

        # 2. Combine all content from Tavily results
        combined_content = ""
        url_favicon_map = {}
        
        for result in crawl_result["results"]:
            raw_content = result.get("raw_content")
            url = result.get("url", "")
            favicon = result.get("favicon", "")
            
            if raw_content:
                combined_content += raw_content + "\n\n"
                url_favicon_map[url] = favicon
        
        if not combined_content.strip():
            raise HTTPException(status_code=400, detail="No content found to vectorize")
        
        # 3. Split content into chunks with overlap
        # chunk_size=1000 tokens ≈ 4000 characters
        # overlap=200 tokens ≈ 800 characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,           # ~1000 tokens
            chunk_overlap=800,         # ~200 tokens overlap
            separators=["\n\n", "\n", ".", " ", ""],  # Smart separators
            length_function=len
        )
        
        chunks = text_splitter.split_text(combined_content)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from content")
        
        print(f"✓ {len(chunks)} chunks générés avec overlap (size=4000, overlap=800)")
        
        # 4. Create documents from chunks with metadata
        documents = []
        primary_url = body.url
        primary_favicon = url_favicon_map.get(primary_url, "")
        
        for chunk_index, chunk_content in enumerate(chunks):
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "url": primary_url,
                    "favicon": primary_favicon,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk_content)
                }
            )
            documents.append(doc)
        
        print(f"✓ {len(documents)} documents créés avec métadonnées de chunks")

        # 5. Initialize OpenAI client for embeddings (direct API)
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 6. Connect to PostgreSQL and create embeddings
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        collection_name = os.getenv("DOCUMENTS_COLLECTION", "crawled_documents")
        
        # Modifier la table existante pour changer collection_id de UUID à TEXT
        # D'abord, vérifier si la table existe et si la colonne est de type UUID
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'langchain_pg_embedding' 
            AND column_name = 'collection_id'
        """)
        column_info = cursor.fetchone()
        
        if column_info and column_info[1] == 'uuid':
            # La colonne existe et est de type UUID, on la modifie en TEXT
            print("⚠️  Modification de la colonne collection_id (UUID → TEXT)...")
            cursor.execute("""
                ALTER TABLE langchain_pg_embedding 
                ALTER COLUMN collection_id TYPE TEXT 
                USING collection_id::TEXT
            """)
            conn.commit()
            print("Colonne collection_id modifiée en TEXT")
        
        # Create table if not exists (avec collection_id en TEXT)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding VECTOR(2000),
                document TEXT,
                cmetadata JSONB
            )
        """)
        
        # Créer un index pour optimiser les recherches de similarité si pas existant
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
            ON langchain_pg_embedding 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        conn.commit()
        
        # 7. Generate embeddings and store in PGVector
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        for i, (doc, doc_id) in enumerate(zip(documents, uuids)):
            # Generate embedding using direct OpenAI API
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=doc.page_content,
                dimensions=2000
            )
            embedding = response.data[0].embedding
            
            # Store in PGVector avec collection_name en TEXT
            cursor.execute("""
                INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    document = EXCLUDED.document, 
                    cmetadata = EXCLUDED.cmetadata
            """, (doc_id, collection_name, embedding, doc.page_content, json.dumps(doc.metadata)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ {len(documents)} documents vectorisés et stockés dans PGVector")

        return JSONResponse(
            content={
                "success": True,
                "message": f"Successfully vectorized {len(documents)} chunks from {body.url}",
                "documents_count": len(documents),
                "chunks_info": {
                    "chunk_size": 4000,
                    "chunk_overlap": 800,
                    "total_chunks": len(chunks)
                }
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error vectorizing URL: {str(e)}")


# ============================================================================
# ENDPOINTS DE GESTION DES CONVERSATIONS
# ============================================================================

@app.get("/conversations")
async def list_conversations(limit: int = 20, offset: int = 0):
    """
    Liste toutes les conversations avec pagination
    
    Query params:
        limit: Nombre max de conversations (défaut: 20)
        offset: Offset pour pagination (défaut: 0)
    
    Returns:
        Liste des conversations avec metadata
    """
    try:
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                thread_id, 
                title, 
                created_at, 
                updated_at, 
                message_count,
                last_message_preview
            FROM conversation_metadata
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "thread_id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
                "message_count": row[4],
                "last_message_preview": row[5]
            })
        
        cursor.close()
        conn.close()
        
        return {"conversations": conversations, "total": len(conversations)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur liste conversations: {str(e)}")


@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """
    Récupère les détails d'une conversation spécifique
    
    Args:
        thread_id: ID de la conversation
    
    Returns:
        Historique complet des messages
    """
    try:
        agent_graph = get_crag_graph()
        
        # Récupérer le dernier checkpoint via le checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        
        # Utiliser get_state pour récupérer l'état complet
        state_snapshot = agent_graph.get_state(config)
        
        if not state_snapshot or not state_snapshot.values:
            raise HTTPException(status_code=404, detail="Conversation non trouvée")
        
        # Extraire les messages
        messages = state_snapshot.values.get("messages", [])
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
                "additional_kwargs": getattr(msg, "additional_kwargs", {})
            })
        
        return {
            "thread_id": thread_id,
            "messages": formatted_messages,
            "message_count": len(formatted_messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération conversation: {str(e)}")


@app.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """
    Supprime une conversation et toutes ses données
    
    Args:
        thread_id: ID de la conversation à supprimer
    
    Returns:
        Message de confirmation
    """
    try:
        conn = psycopg2.connect(postgres_connection_string)
        cursor = conn.cursor()
        
        # Supprimer des tables checkpoints
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        deleted_checkpoints = cursor.rowcount
        
        cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        deleted_writes = cursor.rowcount
        
        cursor.execute("DELETE FROM conversation_metadata WHERE thread_id = %s", (thread_id,))
        deleted_metadata = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": f"Conversation {thread_id} supprimée",
            "deleted": {
                "checkpoints": deleted_checkpoints,
                "writes": deleted_writes,
                "metadata": deleted_metadata
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur suppression conversation: {str(e)}")


# ============================================================================
# ENDPOINTS CRAG/RAG
# ============================================================================

@app.post("/crag/query")
async def crag_query(
    body: CragQueryRequest,
    fastapi_request: Request,
):
    """
    Endpoint pour tester le workflow Hybrid RAG complet avec mémoire conversationnelle
    
    Le workflow Hybrid RAG:
    1. ROUTE_QUESTION: Classifie la question (casual vs admin)
    2. CASUAL_CONVO: Réponses amicales pour conversations informelles
    3. AGENT_RAG: Agent ReAct pour questions administratives
    
    Args:
        body: CragQueryRequest avec question et conversation_id optionnel
        
    Returns:
        JSON avec la réponse générée et des métadonnées du workflow
    """
    try:
        # Générer un conversation_id si non fourni (utiliser thread_id pour LangGraph)
        thread_id = body.conversation_id or str(uuid4())
        
        print(f"\n{'='*60}")
        print(f"Hybrid RAG Query Request")
        print(f"{'='*60}")
        print(f"Question: {body.question}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*60}\n")
        
        # Récupérer le graph Agent RAG (avec InMemorySaver intégré)
        agent_graph = get_crag_graph()
        
        # Préparer l'état initial avec MessagesState
        # On crée un HumanMessage avec la question
        initial_state = {
            "messages": [HumanMessage(content=body.question)],
            "question": body.question,
            "domain_validated": False
        }
        
        # Configuration pour le checkpointer (thread_id pour la mémoire)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Exécuter le workflow Agent RAG avec persistance de la mémoire (SYNC avec InMemorySaver)
        final_state = agent_graph.invoke(initial_state, config)
        
        print(f"\n{'='*60}")
        print(f"Hybrid RAG Workflow Completed")
        print(f"{'='*60}")
        print(f"Messages: {len(final_state.get('messages', []))}")
        print(f"{'='*60}\n")
        
        # Extraire la réponse finale des messages
        messages = final_state.get("messages", [])
        final_answer = ""
        sources = []
        
        # Trouver le dernier AIMessage et extraire sources
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                final_answer = msg.content
                # Extraire les sources des additional_kwargs si présentes
                if hasattr(msg, 'additional_kwargs'):
                    sources = msg.additional_kwargs.get("sources", [])
                break
        
        # Si pas de réponse trouvée dans les messages, essayer l'ancien format
        if not final_answer:
            final_answer = final_state.get("response", "Aucune réponse générée")
        
        response_data = {
            "success": True,
            "conversation_id": thread_id,
            "question": body.question,
            "answer": final_answer,
            "sources": sources,  # Liste complète des sources avec URLs
            "metadata": {
                "workflow": "hybrid_rag",
                "messages_count": len(messages),
                "sources_count": len(sources)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Erreur dans Agent RAG workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error in Agent RAG workflow: {str(e)}"
        )


@app.post("/crag/stream")
async def crag_stream(
    body: CragQueryRequest,
    fastapi_request: Request,
):
    """
    Endpoint Hybrid RAG avec streaming en temps réel (Server-Sent Events).
    
    Version streaming de /crag/query qui permet de suivre l'exécution
    du workflow node par node et de recevoir la réponse token par token.
    
    Le workflow Hybrid RAG:
    1. ROUTE_QUESTION: Classifie la question (casual vs admin)
    2. CASUAL_CONVO: Réponses amicales pour conversations informelles  
    3. AGENT_RAG: Agent ReAct pour questions administratives
    
    Args:
        body: CragQueryRequest avec question et conversation_id optionnel
        
    Returns:
        StreamingResponse avec events SSE (Server-Sent Events)
        
    Format des events:
        - {"type": "node_start", "node": "route_question"}
        - {"type": "node_end", "node": "route_question", "question_type": "casual|admin"}
        - {"type": "node_start", "node": "casual_convo"}
        - {"type": "message_chunk", "content": "...", "node": "casual_convo"}
        - {"type": "node_start", "node": "agent_rag"}
        - {"type": "message_chunk", "content": "...", "node": "agent_rag"}
        - {"type": "complete", "conversation_id": "...", "answer": "...", "sources": [...]}
    """
    # No Authorization header required for streaming endpoint
    
    async def event_generator():
        """Génère les events SSE pour le streaming."""
        try:
            # Générer un conversation_id si non fourni
            thread_id = body.conversation_id or str(uuid4())
            
            print(f"\n{'='*60}")
            print(f"Hybrid RAG Stream Request")
            print(f"{'='*60}")
            print(f"Question: {body.question}")
            print(f"Thread ID: {thread_id}")
            print(f"{'='*60}\n")
            
            # Récupérer le graph Agent RAG
            agent_graph = get_crag_graph()
            
            # ✅ IMPORTANT : Récupérer l'historique existant du checkpoint
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = agent_graph.get_state(config)
            
            # Charger les anciens messages s'il y en a
            old_messages = []
            if state_snapshot and state_snapshot.values:
                old_messages = state_snapshot.values.get("messages", [])
                print(f"✓ Historique chargé : {len(old_messages)} messages précédents")
            else:
                print(f"✓ Nouvelle conversation créée")
            
            # Ajouter le nouveau message à l'historique
            new_message = HumanMessage(content=body.question)
            all_messages = old_messages + [new_message]
            
            # Préparer l'état initial avec l'historique COMPLET
            initial_state = {
                "messages": all_messages,  # ← Historique + nouveau message
                "question": body.question,
                "domain_validated": False
            }
            
            # Variables pour accumuler la réponse et les sources
            accumulated_answer = ""
            collected_sources = []
            
            # Streamer le workflow Agent RAG en mode SYNCHRONE (PostgresSaver n'est pas async)
            # Utiliser stream() au lieu de astream() et wrapper dans to_thread
            def sync_stream():
                """Fonction synchrone pour streamer le graph"""
                for event in agent_graph.stream(initial_state, config):
                    yield event
            
            # Exécuter le stream synchrone dans un thread séparé
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            executor = ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            
            # Wrapper pour itérer sur le stream synchrone de manière async
            async def async_event_iterator():
                stream_gen = sync_stream()
                while True:
                    try:
                        event = await loop.run_in_executor(executor, next, stream_gen, None)
                        if event is None:
                            break
                        yield event
                    except StopIteration:
                        break
            
            async for event in async_event_iterator():
                # event est un dict avec une clé = nom du node
                # et valeur = état retourné par ce node
                
                for node_name, node_output in event.items():
                    print(f"Node: {node_name}")
                    
                    # ─────────────────────────────────────────────────
                    # ROUTE_QUESTION node
                    # ─────────────────────────────────────────────────
                    if node_name == "route_question":
                        question_type = node_output.get("question_type", "admin")
                        
                        # Émettre un status pour route_question
                        yield (
                            json.dumps({
                                "type": "status",
                                "step": "route_question",
                                "message": "Classification de la question..."
                            }) + "\n"
                        )
                        
                        yield (
                            json.dumps({
                                "type": "node_start",
                                "node": "route_question",
                                "message": "Classification de la question..."
                            }) + "\n"
                        )
                        
                        yield (
                            json.dumps({
                                "type": "node_end",
                                "node": "route_question",
                                "question_type": question_type,
                                "message": f"Question classifiée: {question_type}"
                            }) + "\n"
                        )
                    
                    # ─────────────────────────────────────────────────
                    # CASUAL_CONVO node
                    # ─────────────────────────────────────────────────
                    elif node_name == "casual_convo":
                        # Émettre un status pour casual_convo
                        yield (
                            json.dumps({
                                "type": "status",
                                "step": "casual_convo",
                                "message": "Conversation informelle..."
                            }) + "\n"
                        )
                        
                        yield (
                            json.dumps({
                                "type": "node_start",
                                "node": "casual_convo",
                                "message": "Génération de réponse conversationnelle..."
                            }) + "\n"
                        )
                        
                        # Extraire la réponse du dernier AIMessage
                        messages = node_output.get("messages", [])
                        
                        for msg in reversed(messages):
                            if hasattr(msg, 'type') and msg.type == 'ai':
                                accumulated_answer = msg.content
                                break
                        
                        # Streaming caractère par caractère pour réponses casual
                        for char in accumulated_answer:
                            yield (
                                json.dumps({
                                    "type": "message_chunk",
                                    "content": char,
                                    "node": "casual_convo"
                                }) + "\n"
                            )
                            await asyncio.sleep(0.005)  # 5ms entre chaque caractère
                        
                        yield (
                            json.dumps({
                                "type": "node_end",
                                "node": "casual_convo",
                                "message": f"Réponse conversationnelle générée ({len(accumulated_answer)} caractères)"
                            }) + "\n"
                        )
                    
                    # ─────────────────────────────────────────────────
                    # AGENT_RAG node
                    # ─────────────────────────────────────────────────
                    elif node_name == "agent_rag":
                        # Émettre un status pour agent_rag
                        yield (
                            json.dumps({
                                "type": "status",
                                "step": "agent_rag",
                                "message": "Agent ReAct en cours..."
                            }) + "\n"
                        )
                        
                        yield (
                            json.dumps({
                                "type": "node_start",
                                "node": "agent_rag",
                                "message": "Agent ReAct en cours d'exécution..."
                            }) + "\n"
                        )
                        
                        # Extraire la réponse et les sources du dernier AIMessage
                        messages = node_output.get("messages", [])
                        
                        for msg in reversed(messages):
                            if hasattr(msg, 'type') and msg.type == 'ai':
                                accumulated_answer = msg.content
                                
                                # Extraire les sources des additional_kwargs
                                if hasattr(msg, 'additional_kwargs'):
                                    collected_sources = msg.additional_kwargs.get("sources", [])
                                
                                break
                        
                        # Détecter les tools utilisés dans la réponse de l'agent pour émettre des status
                        vector_search_used = False
                        web_search_used = False
                        
                        # Analyser les sources pour détecter les outils utilisés
                        for source in collected_sources:
                            source_type = source.get("type", "")
                            if source_type == "vector_search" or "similarity_score" in source:
                                if not vector_search_used:
                                    vector_search_used = True
                                    yield (
                                        json.dumps({
                                            "type": "status",
                                            "step": "vector_search",
                                            "message": "Recherche vectorielle en cours..."
                                        }) + "\n"
                                    )
                            elif source_type == "web_search" or "web" in source.get("url", "").lower():
                                if not web_search_used:
                                    web_search_used = True
                                    yield (
                                        json.dumps({
                                            "type": "status",
                                            "step": "web_search",
                                            "message": "Recherche web en cours..."
                                        }) + "\n"
                                    )
                        
                        # Émettre status pour la génération de la réponse
                        yield (
                            json.dumps({
                                "type": "status",
                                "step": "generate",
                                "message": "Génération de la réponse..."
                            }) + "\n"
                        )
                        
                        # Streaming caractère par caractère pour une expérience fluide
                        for char in accumulated_answer:
                            yield (
                                json.dumps({
                                    "type": "message_chunk",
                                    "content": char,
                                    "node": "agent_rag"
                                }) + "\n"
                            )
                            # Petit délai pour un streaming plus naturel
                            await asyncio.sleep(0.005)  # 5ms entre chaque caractère
                        
                        yield (
                            json.dumps({
                                "type": "node_end",
                                "node": "agent_rag",
                                "message": f"Réponse générée ({len(accumulated_answer)} caractères, {len(collected_sources)} sources)"
                            }) + "\n"
                        )
            
            # ─────────────────────────────────────────────────────────
            # EVENT FINAL - Workflow complet
            # ─────────────────────────────────────────────────────────
            print(f"\n{'='*60}")
            print(f"Hybrid RAG Stream Completed")
            print(f"{'='*60}")
            print(f"Réponse: {len(accumulated_answer)} caractères")
            print(f"Sources: {len(collected_sources)}")
            print(f"{'='*60}\n")
            
            # ─────────────────────────────────────────────────────────
            # LOGGING DE LA CONVERSATION (PostgreSQL)
            # ─────────────────────────────────────────────────────────
            try:
                conn = psycopg2.connect(postgres_connection_string)
                cursor = conn.cursor()
                
                # Déterminer les tools utilisés basé sur les sources
                tools_used = []
                vector_searches = 0
                web_searches = 0
                
                for source in collected_sources:
                    source_type = source.get("type", "")
                    if source_type == "vector_search" or "similarity_score" in source:
                        if "vector_search" not in tools_used:
                            tools_used.append("vector_search")
                        vector_searches += 1
                    elif source_type == "web_search" or "web" in source.get("url", "").lower():
                        if "web_search" not in tools_used:
                            tools_used.append("web_search")
                        web_searches += 1
                
                if collected_sources and "reranker" not in tools_used:
                    tools_used.append("reranker")
                
                # Insérer dans la table conversations
                cursor.execute("""
                    INSERT INTO conversations (
                        id, question, answer, sources, tools_used,
                        vector_searches, web_searches, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        answer = EXCLUDED.answer,
                        sources = EXCLUDED.sources,
                        tools_used = EXCLUDED.tools_used,
                        vector_searches = EXCLUDED.vector_searches,
                        web_searches = EXCLUDED.web_searches,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                """, (
                    thread_id,
                    body.question,
                    accumulated_answer,
                    json.dumps(collected_sources),
                    json.dumps(tools_used),  # Convertir la liste en JSON
                    vector_searches,
                    web_searches,
                    "completed"
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                print(f"💾 Conversation {thread_id} enregistrée dans PostgreSQL")
                
                # ─────────────────────────────────────────────────────────
                # LOGGING METADATA CONVERSATION (pour UI/gestion)
                # ─────────────────────────────────────────────────────────
                try:
                    conn = psycopg2.connect(postgres_connection_string)
                    cursor = conn.cursor()
                    
                    # Générer titre automatique (premiers 50 caractères de la question)
                    title = body.question[:50] + "..." if len(body.question) > 50 else body.question
                    
                    # Prévisualisation de la dernière réponse (premiers 100 caractères)
                    last_message_preview = accumulated_answer[:100] + "..." if len(accumulated_answer) > 100 else accumulated_answer
                    
                    # Upsert dans conversation_metadata
                    cursor.execute("""
                        INSERT INTO conversation_metadata 
                            (thread_id, title, message_count, last_message_preview, updated_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (thread_id) DO UPDATE SET
                            message_count = conversation_metadata.message_count + 2,
                            last_message_preview = EXCLUDED.last_message_preview,
                            updated_at = NOW()
                    """, (
                        thread_id,
                        title,
                        2,  # Question + réponse = 2 messages
                        last_message_preview
                    ))
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    print(f"💾 Metadata conversation {thread_id} enregistrée")
                    
                except Exception as metadata_error:
                    print(f"⚠️ Erreur logging metadata: {str(metadata_error)}")
                
            except Exception as log_error:
                print(f"⚠️ Erreur lors de l'enregistrement de la conversation: {str(log_error)}")
                # Ne pas bloquer le stream en cas d'erreur de logging
            
            yield (
                json.dumps({
                    "type": "complete",
                    "conversation_id": thread_id,
                    "question": body.question,
                    "answer": accumulated_answer,
                    "sources": collected_sources,
                    "metadata": {
                        "workflow": "hybrid_rag",
                        "sources_count": len(collected_sources),
                        "answer_length": len(accumulated_answer)
                    }
                }) + "\n"
            )
            
        except Exception as e:
            print(f"Erreur dans CRAG stream: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Logger l'erreur dans la base de données
            try:
                conn = psycopg2.connect(postgres_connection_string)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO conversations (
                        id, question, status, error_message
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        error_message = EXCLUDED.error_message,
                        updated_at = NOW()
                """, (
                    thread_id,
                    body.question,
                    "error",
                    str(e)
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                print(f"💾 Erreur de conversation {thread_id} enregistrée dans PostgreSQL")
                
            except Exception as log_error:
                print(f"⚠️ Erreur lors de l'enregistrement de l'erreur: {str(log_error)}")
            
            yield (
                json.dumps({
                    "type": "error",
                    "error": str(e),
                    "message": f"Erreur: {str(e)}"
                }) + "\n"
            )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Désactive le buffering nginx
        }
    )



#============================================================================
# ENDPOINTS POUR LA GESTION FILE UPLOA VIA la PAGE admin
# ============================================================================


@app.post("/api/upload")
async def upload_pdf_file(file: UploadFile = File(...)):
    """
    Reçoit et sauvegarde un fichier PDF dans le répertoire 'uploads'.
    """
    # Vérifie si le nom du fichier se termine bien par .pdf
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Uniquement les .pdf sont acceptés."
        )

    try:
        # Définit le nom du dossier où seront stockés les fichiers
        uploads_dir = Path("uploads")
        # Crée le dossier s'il n'existe pas déjà
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Définit le chemin complet où le fichier sera sauvegardé
        file_path = uploads_dir / file.filename
        
        # Lit le contenu du fichier envoyé
        contents = await file.read()
        # Écrit le contenu dans le nouveau fichier sur le serveur
        with open(file_path, "wb") as f:
            f.write(contents)

        print(f"✓ Fichier '{file.filename}' sauvegardé avec succès dans '{file_path}'")

        # Renvoie une réponse JSON pour confirmer le succès
        return JSONResponse(
            content={
                "success": True,
                "message": f"Fichier '{file.filename}' envoyé avec succès.",
                "filename": file.filename,
                "saved_path": str(file_path)
            }
        )

    except Exception as e:
        # En cas d'erreur pendant la sauvegarde, renvoie une erreur 500
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de la sauvegarde du fichier: {str(e)}"
        )


# ============================================================================
# NOTE : Endpoints /conversations/* et /sources/* SUPPRIMÉS
# Pas de persistence Supabase - Seulement InMemorySaver pour mémoire volatile
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
