import { useState, useCallback, useRef, useEffect } from 'react';

// ============================================================================
// TYPES
// ============================================================================

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  metadata?: MessageMetadata;
  timestamp?: string;
}

interface Source {
  url?: string;
  content?: string;
  similarity_score?: number;
  rerank_score?: number;
  type?: string;
}

interface MessageMetadata {
  workflow?: string;
  sources_count?: number;
  answer_length?: number;
}

interface StatusUpdate {
  step: 'route_question' | 'vector_search' | 'web_search' | 'generate' | 'casual_convo' | 'agent_rag';
  message: string;
}

interface SSEEvent {
  type: 'status' | 'node_start' | 'node_end' | 'message_chunk' | 'complete' | 'error';
  step?: string;
  message?: string;
  content?: string;
  node?: string;
  question_type?: string;
  answer?: string;
  sources?: Source[];
  metadata?: MessageMetadata;
  conversation_id?: string;
  error?: string;
}

// ============================================================================
// HOOK: useHybridRAG
// ============================================================================

/**
 * Hook personnalisé pour gérer les conversations avec le système Hybrid RAG
 * 
 * Fonctionnalités :
 * - Gestion automatique du conversation_id (localStorage)
 * - Streaming SSE pour réponses en temps réel
 * - Status updates (routing, vector search, web search, generation)
 * - Pas d'envoi d'historique (géré par PostgresSaver backend)
 * - Accumulation des messages pour affichage UI
 * 
 * @returns {Object} - messages, submitQuery, resetConversation, isLoading, conversationId, currentStatus
 */
export function useHybridRAG() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStatus, setCurrentStatus] = useState<StatusUpdate | null>(null);
  
  // Conversation ID persistant (localStorage) - REACTIF
  const [conversationId, setConversationId] = useState<string>(() => {
    const stored = localStorage.getItem('dagan_conversation_id');
    if (stored) return stored;
    
    const newId = crypto.randomUUID();
    localStorage.setItem('dagan_conversation_id', newId);
    return newId;
  });

  // Référence pour éviter les double-appels
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Charger l'historique des messages au montage du composant
   */
  useEffect(() => {
    const loadConversationHistory = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
        const response = await fetch(
          `${apiUrl}/conversations/${conversationId}`,
          { method: 'GET' }
        );

        if (response.ok) {
          const data = await response.json();
          // data.messages contient l'historique du backend
          if (Array.isArray(data.messages)) {
            setMessages(data.messages);
          }
        }
      } catch (error) {
        console.error('Erreur lors du chargement de l\'historique:', error);
        // Continuer même si le chargement échoue
      }
    };

    // Charger l'historique au montage (une seule fois)
    loadConversationHistory();
  }, [conversationId]); // Recharger si conversationId change

  /**
   * Soumet une question au backend Hybrid RAG avec streaming SSE
   * 
   * @param question - Question de l'utilisateur
   */
  const submitQuery = useCallback(async (question: string) => {
    if (!question.trim()) return;
    
    // Annuler toute requête en cours
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    abortControllerRef.current = new AbortController();
    setIsLoading(true);
    // Initialiser avec le status "route_question" (phase de reflexion)
    setCurrentStatus({
      step: 'route_question',
      message: 'Analyse de votre question...'
    });
    
    // Ajouter question à l'UI immédiatement
    const userMessage: Message = {
      role: 'user',
      content: question,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      // Appel API streaming (SANS historique, juste question + conversation_id)
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/crag/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          conversation_id: conversationId  // ← Seulement l'ID, pas d'historique !
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Pas de reader disponible');
      }

      const decoder = new TextDecoder();
      let accumulatedContent = '';
      
      // Message assistant temporaire (vide)
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Lire le stream SSE
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(l => l.trim());

        for (const line of lines) {
          try {
            const event: SSEEvent = JSON.parse(line);

            // ──────────────────────────────────────────────
            // STATUS UPDATE (vector_search, web_search, etc.)
            // ──────────────────────────────────────────────
            if (event.type === 'status') {
              setCurrentStatus({
                step: event.step as StatusUpdate['step'],
                message: event.message || ''
              });
            }
            
            // ──────────────────────────────────────────────
            // NODE START/END (route_question, agent_rag, etc.)
            // ──────────────────────────────────────────────
            else if (event.type === 'node_start') {
              setCurrentStatus({
                step: event.node as StatusUpdate['step'],
                message: event.message || `Démarrage: ${event.node}`
              });
            }
            else if (event.type === 'node_end') {
              // Ne pas clear le status, le prochain status le remplacera
              // On clear seulement sur 'complete'
            }
            
            // ──────────────────────────────────────────────
            // MESSAGE CHUNK (streaming caractère par caractère)
            // ──────────────────────────────────────────────
            else if (event.type === 'message_chunk') {
              accumulatedContent += event.content || '';
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: accumulatedContent
                };
                return updated;
              });
            }
            
            // ──────────────────────────────────────────────
            // COMPLETE (réponse finale avec sources)
            // ──────────────────────────────────────────────
            else if (event.type === 'complete') {
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: 'assistant',
                  content: event.answer || accumulatedContent,
                  sources: event.sources,
                  metadata: event.metadata,
                  timestamp: new Date().toISOString()
                };
                return updated;
              });
              setCurrentStatus(null);
            }
            
            // ──────────────────────────────────────────────
            // ERROR
            // ──────────────────────────────────────────────
            else if (event.type === 'error') {
              console.error('Erreur SSE:', event.error);
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: 'assistant',
                  content: `Erreur: ${event.error || 'Une erreur est survenue'}`,
                  timestamp: new Date().toISOString()
                };
                return updated;
              });
              setCurrentStatus(null);
            }
          } catch (parseError) {
            console.error('Erreur parsing SSE event:', parseError, line);
          }
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('Requête annulée');
        return;
      }
      
      console.error('Erreur submitQuery:', error);
      setMessages(prev => {
        // Remplacer le dernier message (vide) par un message d'erreur
        const updated = [...prev];
        if (updated[updated.length - 1]?.role === 'assistant') {
          updated[updated.length - 1] = {
            role: 'assistant',
            content: 'Désolé, une erreur est survenue. Veuillez réessayer.',
            timestamp: new Date().toISOString()
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
      setCurrentStatus(null);
      abortControllerRef.current = null;
    }
  }, [conversationId]);

  /**
   * Réinitialise la conversation (nouveau conversation_id + clear messages)
   */
  const resetConversation = useCallback(() => {
    // Annuler requête en cours si existante
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Générer nouveau conversation_id
    const newId = crypto.randomUUID();
    localStorage.setItem('dagan_conversation_id', newId);
    setConversationId(newId);  // ← Met à jour l'état React
    
    // Clear messages
    setMessages([]);
    setCurrentStatus(null);
    setIsLoading(false);
    
    // ✓ Plus besoin de reload, React va recharger les messages via le useEffect
  }, []);

  return {
    messages,
    submitQuery,
    resetConversation,
    isLoading,
    conversationId,
    currentStatus
  };
}
