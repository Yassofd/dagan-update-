-- Script d'initialisation de la base de données Dagan
-- Version simplifiée pour InMemorySaver (mémoire volatile)
-- Seules 2 tables essentielles : langchain_pg_embedding et conversations

-- Activer l'extension PGVector pour les embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================
-- Table: langchain_pg_embedding
-- Pour stocker les documents avec leurs embeddings
-- =============================
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    embedding VECTOR(2000),
    document TEXT NOT NULL,
    cmetadata JSONB DEFAULT '{}'::jsonb
);

-- Index pour recherche vectorielle
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx
ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index pour filtrage par collection
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_idx
ON langchain_pg_embedding (collection_id);

-- =============================
-- Table: conversations
-- Pour stocker l'historique des conversations (monitoring uniquement)
-- =============================
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    title VARCHAR(500),
    question TEXT,
    answer TEXT,
    sources JSONB DEFAULT '[]'::jsonb,
    tools_used JSONB DEFAULT '[]'::jsonb,
    vector_searches INTEGER DEFAULT 0,
    web_searches INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index pour recherche par utilisateur
CREATE INDEX IF NOT EXISTS idx_conversations_user
ON conversations(user_id);

-- Index pour tri par date
CREATE INDEX IF NOT EXISTS idx_conversations_created
ON conversations(created_at DESC);

-- Index pour recherche par statut
CREATE INDEX IF NOT EXISTS idx_conversations_status
ON conversations(status);

-- =============================
-- Fonction: update_updated_at
-- Pour mettre à jour automatiquement updated_at
-- =============================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger pour conversations
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================
-- TABLES POUR LANGGRAPH CHECKPOINTER (PostgresSaver)
-- Pour la persistance de la mémoire conversationnelle
-- =============================

-- Table principale des checkpoints
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Index pour améliorer les performances de recherche
CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx 
    ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS checkpoints_parent_id_idx 
    ON checkpoints(parent_checkpoint_id);

-- Table pour les writes (événements intermédiaires)
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Table metadata des conversations (pour UI/gestion)
CREATE TABLE IF NOT EXISTS conversation_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT UNIQUE NOT NULL,
    title TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    last_message_preview TEXT
);

-- Index pour recherche et tri
CREATE INDEX IF NOT EXISTS conversation_metadata_thread_id_idx 
    ON conversation_metadata(thread_id);
CREATE INDEX IF NOT EXISTS conversation_metadata_updated_at_idx 
    ON conversation_metadata(updated_at DESC);

-- Trigger pour conversation_metadata
DROP TRIGGER IF EXISTS update_conversation_metadata_updated_at ON conversation_metadata;
CREATE TRIGGER update_conversation_metadata_updated_at
    BEFORE UPDATE ON conversation_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================
-- Affichage des tables créées
-- =============================
DO $$
BEGIN
    RAISE NOTICE 'Base de données Dagan initialisée avec succès !';
    RAISE NOTICE 'Tables créées:';
    RAISE NOTICE '   - langchain_pg_embedding (embeddings vectoriels)';
    RAISE NOTICE '   - conversations (historique des conversations)';
    RAISE NOTICE '   - checkpoints (LangGraph checkpointer)';
    RAISE NOTICE '   - checkpoint_writes (événements LangGraph)';
    RAISE NOTICE '   - conversation_metadata (métadonnées conversations)';
    RAISE NOTICE 'Extension PGVector activée pour recherche vectorielle';
    RAISE NOTICE 'Mémoire conversationnelle: PostgresSaver (persistante)';
    RAISE NOTICE 'Prêt à l''utilisation !';
END $$;
