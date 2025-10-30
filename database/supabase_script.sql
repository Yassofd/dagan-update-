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

-- =============================
-- Affichage des tables créées
-- =============================
DO $$
BEGIN
    RAISE NOTICE 'Base de données Dagan initialisée avec succès !';
    RAISE NOTICE 'ables créées:';
    RAISE NOTICE '   - langchain_pg_embedding (embeddings vectoriels)';
    RAISE NOTICE '   - conversations (historique des conversations)';
    RAISE NOTICE ' Extension PGVector activée pour recherche vectorielle';
    RAISE NOTICE 'Mémoire conversationnelle: InMemorySaver (volatile)';
    RAISE NOTICE 'Prêt à l''utilisation !';
END $$;
