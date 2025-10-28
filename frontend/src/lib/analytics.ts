/**
 * Google Analytics Helper
 * 
 * Fonctions utilitaires pour tracker les événements dans Google Analytics 4
 */

declare global {
  interface Window {
    gtag?: (
      command: string,
      eventNameOrConfig: string,
      params?: Record<string, any>
    ) => void;
  }
}

/**
 * Vérifie si Google Analytics est chargé
 */
export const isGALoaded = (): boolean => {
  return typeof window !== 'undefined' && typeof window.gtag === 'function';
};

/**
 * Track une page vue
 * @param path - Chemin de la page (ex: '/chat', '/about')
 */
export const trackPageView = (path: string): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'page_view', {
      page_path: path,
    });
  }
};

/**
 * Track une question posée par l'utilisateur
 * @param questionLength - Longueur de la question
 * @param conversationId - ID de la conversation
 */
export const trackQuestion = (
  questionLength: number,
  conversationId: string | null
): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'question_sent', {
      question_length: questionLength,
      has_conversation_id: !!conversationId,
      event_category: 'engagement',
      event_label: 'chat_interaction',
    });
  }
};

/**
 * Track une réponse reçue du bot
 * @param responseLength - Longueur de la réponse
 * @param sourcesCount - Nombre de sources utilisées
 * @param conversationId - ID de la conversation
 */
export const trackResponse = (
  responseLength: number,
  sourcesCount: number,
  conversationId: string | null
): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'response_received', {
      response_length: responseLength,
      sources_count: sourcesCount,
      has_conversation_id: !!conversationId,
      event_category: 'engagement',
      event_label: 'bot_response',
    });
  }
};

/**
 * Track un clic sur une source
 * @param sourceUrl - URL de la source cliquée
 * @param sourceTitle - Titre de la source
 */
export const trackSourceClick = (sourceUrl: string, sourceTitle?: string): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'source_click', {
      source_url: sourceUrl,
      source_title: sourceTitle || 'Unknown',
      event_category: 'engagement',
      event_label: 'external_link',
    });
  }
};

/**
 * Track une nouvelle conversation
 */
export const trackNewConversation = (): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'new_conversation', {
      event_category: 'engagement',
      event_label: 'conversation_reset',
    });
  }
};

/**
 * Track un clic sur une question suggérée
 * @param question - Texte de la question suggérée
 */
export const trackSuggestedQuestionClick = (question: string): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'suggested_question_click', {
      question_text: question,
      event_category: 'engagement',
      event_label: 'quick_action',
    });
  }
};

/**
 * Track une erreur
 * @param errorMessage - Message d'erreur
 * @param errorType - Type d'erreur (ex: 'network', 'api', 'validation')
 */
export const trackError = (errorMessage: string, errorType: string): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'error_occurred', {
      error_message: errorMessage,
      error_type: errorType,
      event_category: 'errors',
    });
  }
};

/**
 * Track le temps passé sur une conversation
 * @param durationSeconds - Durée en secondes
 */
export const trackConversationDuration = (durationSeconds: number): void => {
  if (isGALoaded()) {
    window.gtag!('event', 'conversation_duration', {
      duration_seconds: durationSeconds,
      event_category: 'engagement',
      event_label: 'time_spent',
    });
  }
};
