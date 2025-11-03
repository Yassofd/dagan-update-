import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, User, Lightbulb } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { SourcesFavicons, Source } from "./SourcesFavicons";
import { CitationsPanel } from "./CitationsPanel";
import { StreamingMessage } from "./StreamingMessage";
import { ToolPipeline } from "./ToolPipeline";
import { useHybridRAG } from "@/hooks/useHybridRAG";
import { trackQuestion, trackResponse, trackNewConversation, trackSuggestedQuestionClick } from "@/lib/analytics";
import avatarImage from "@/assets/avatar.svg";
import reflexionImage from "@/assets/reflexion.svg";
import logoImage from "@/assets/Novatekis.svg";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface ToolStep {
  type: "validate_domain" | "agent_rag" | "vector_search" | "web_search" | "generate";
  status: "pending" | "active" | "completed";
  count?: number;
  details?: string;
}

const SUGGESTED_QUESTIONS = [
  "Comment renouveler mon passeport ?",
  "Comment obtenir un duplicata d'un certificat de nationalité ?",
  "Ai-je besoin d'un visa pour aller en Malaisie?",
  "Quelle est la procédure pour étudier en France pour un étudiant Togolais ?"
];

export const ChatInterface = () => {
  // Utiliser le hook useHybridRAG pour la gestion des messages et du streaming
  const { messages, submitQuery, resetConversation, isLoading, conversationId, currentStatus } = useHybridRAG();
  
  const [input, setInput] = useState("");
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Auto-scroll quand nouveaux messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-scroll smooth pendant le chargement
  useEffect(() => {
    if (isLoading && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const sendMessage = async (text?: string) => {
    const payload = (text ?? input).trim();
    if (!payload || isLoading) return;

    // Track question dans Google Analytics
    trackQuestion(payload.length, conversationId);

    setInput("");
    
    try {
      // Utiliser le hook pour envoyer la question
      await submitQuery(payload);
      
      // Track response (sera appelé après réponse complète dans le hook)
      // On pourrait ajouter un callback au hook si besoin
    } catch (error: any) {
      toast({
        title: "Erreur",
        description: error?.message || "Une erreur est survenue lors de l'envoi du message.",
        variant: "destructive"
      });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestionClick = (question: string) => {
    // Track clic sur question suggérée
    trackSuggestedQuestionClick(question);
    setInput(question);
  };

  const handleClearConversation = () => {
    // Track nouvelle conversation
    trackNewConversation();
    
    // Utiliser la fonction du hook
    resetConversation();
  };

  const handleSourceClick = (sources: Source[]) => {
    setSelectedSources(sources);
    setIsPanelOpen(true);
  };

  // Fonction pour déterminer l'avatar à afficher selon l'état
  const getAvatarForStatus = (status?: { step: string; message: string } | null, isCurrentMessage: boolean = false) => {
    // Si c'est un ancien message (terminé), toujours avatar.svg
    if (!isCurrentMessage) {
      return avatarImage;
    }
    
    // Pour le message en cours de génération :
    // Si on est à l'étape "generate" OU si plus de status (terminé), utiliser avatar.svg
    if (!status || status.step === "generate" || status.step === "agent_rag" || status.step === "casual_convo") {
      return avatarImage;
    }
    
    // Pour les autres étapes (route_question, vector_search, web_search), utiliser reflexion.svg
    return reflexionImage;
  };  return (
  <>
  <Card className={`w-full m-0 max-w-none rounded-none bg-transparent border-0 shadow-none h-full max-h-[80vh] ${isPanelOpen ? 'max-w-[calc(100%-410px)] mr-[390px]' : ''}`}>
      <CardHeader className="flex flex-col sm:flex-row items-center justify-between px-3 sm:px-4 py-2 border-b bg-white gap-2 sm:gap-0">
        <div className="flex items-center gap-1 sm:gap-2">
          <span className="text-xs sm:text-sm text-muted-foreground">Développé par</span>
          <a href="https://novatekis.com" target="_blank" rel="noopener noreferrer" className="hover:opacity-80 transition-opacity">
            <img src={logoImage} alt="Novatekis" className="h-5 sm:h-6 md:h-10 w-auto" />
          </a>
        </div>
        <div className="flex items-center gap-2">
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="sm" disabled={isLoading} className="text-xs px-2 h-8">
                Nouvelle conversation
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Nouvelle conversation</AlertDialogTitle>
                <AlertDialogDescription>
                  Êtes-vous sûr de vouloir commencer une nouvelle conversation ?
                  L'historique actuel sera supprimé et vous n'aurez plus accès à cette conversation.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Annuler</AlertDialogCancel>
                <AlertDialogAction onClick={handleClearConversation}>
                  Continuer
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardHeader>
  <CardContent className="p-0 bg-white h-full flex flex-col">
    <ScrollArea className="flex-1 p-3 sm:p-4" ref={scrollRef}>
          <div className="space-y-3 sm:space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-4 space-y-3">
                <div className="text-center space-y-2">
                  <div className="inline-flex items-center justify-center w-20 h-20 sm:w-28 sm:h-28 mb-0.5">
                    <img src={avatarImage} alt="Dagan Avatar" className="h-16 w-16 sm:h-24 sm:w-24" />
                  </div>
                  <h2 className="text-sm sm:text-lg font-bold text-foreground px-4">Woezon, Bonjour je suis Dagan votre assistant IA</h2>
                  <p className="text-xs text-muted-foreground max-w-xs sm:max-w-md mx-auto px-4">
                    Assistant intelligent pour vos démarches administratives au Togo.
                  </p>
                </div>
                
                <div className="w-full max-w-xs sm:max-w-md md:max-w-lg lg:max-w-2xl space-y-2">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground justify-center">
                    <Lightbulb className="h-3 w-3 text-warning" />
                    <span>Questions suggérées :</span>
                  </div>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2">
                    {SUGGESTED_QUESTIONS.map((question, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSuggestionClick(question)}
                        className="text-left px-3 py-2 rounded-lg border-2 border-primary/20 bg-white hover:bg-highlight hover:border-accent hover:shadow-2xl hover:shadow-accent/50 hover:scale-[1.02] sm:hover:scale-[1.05] transition-all duration-300 text-xs text-foreground font-medium group relative overflow-hidden min-h-[44px] touch-manipulation"
                      >
                        <span className="relative z-10">{question}</span>
                        <div className="absolute inset-0 bg-gradient-to-r from-accent/0 via-accent/30 to-accent/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out"></div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-2 sm:gap-3 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                } animate-fade-in`}
              >
                {message.role === "assistant" && (
                  <div className="h-8 w-8 sm:h-10 sm:w-10 flex items-center justify-center flex-shrink-0">
                    <img 
                      src={getAvatarForStatus(
                        index === messages.length - 1 ? currentStatus : null, 
                        index === messages.length - 1 && isLoading
                      )} 
                      alt="Dagan" 
                      className="h-7 w-7 sm:h-9 sm:w-9" 
                    />
                  </div>
                )}
                
                <div
                  className={`rounded-xl px-3 py-2 max-w-[85%] sm:max-w-[80%] ${
                    message.role === "user"
                      ? "bg-[#86b7b2] text-white"
                      : "bg-white border shadow-sm"
                  }`}
                >
                  <div className="flex flex-col">
                    {message.role === "assistant" ? (
                      <>
                        {/* Afficher l'animation de 3 points si message vide et en cours de chargement */}
                        {isLoading && index === messages.length - 1 && !message.content && (
                          <div className="flex gap-1">
                            <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                            <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                            <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                          </div>
                        )}
                        {/* Afficher le contenu du message */}
                        {message.content && (
                          <StreamingMessage content={message.content} isStreaming={isLoading && index === messages.length - 1} />
                        )}
                      </>
                    ) : (
                      <p className="text-xs leading-relaxed whitespace-pre-wrap">
                        {message.content}
                      </p>
                    )}
                    {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                      <SourcesFavicons sources={message.sources as any} onSourceClick={handleSourceClick} />
                    )}
                  </div>
                </div>
                
                {message.role === "user" && (
                  <div className="h-6 w-6 sm:h-8 sm:w-8 rounded-full bg-primary/15 flex items-center justify-center flex-shrink-0">
                    <User className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                  </div>
                )}
              </div>
            ))}
            
                {/* Afficher l'avatar de chargement seulement si pas de message assistant en cours */}
                {isLoading && !messages.some(m => m.role === "assistant" && messages.indexOf(m) === messages.length - 1) && (
                  <div className="flex gap-2 sm:gap-3 justify-start animate-fade-in">
                    <div className="h-8 w-8 sm:h-10 sm:w-10 flex items-center justify-center flex-shrink-0">
                      <img src={getAvatarForStatus(currentStatus, true)} alt="Dagan" className="h-7 w-7 sm:h-9 sm:w-9" />
                    </div>
                    <div className="rounded-xl px-3 py-2 max-w-[85%] sm:max-w-[80%] bg-white border shadow-sm">
                      <div className="flex gap-1">
                        <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                        <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                        <div className="h-2 w-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
            {/* Référence invisible pour l'auto-scroll */}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
        
        <div className="p-3 sm:p-4 border-t">
          <div className="flex justify-center">
            <div className="relative max-w-3xl w-full">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Posez votre question..."
                disabled={isLoading}
                className="pr-12 text-sm sm:text-xs min-h-[44px] touch-manipulation rounded-full hover:border-accent transition-colors duration-300 focus:ring-0 focus:outline-none focus-visible:ring-0 focus-visible:outline-none"
              />
              <Button 
                onClick={() => sendMessage()} 
                disabled={isLoading || !input.trim()}
                size="icon"
                className="absolute right-1 top-1/2 transform -translate-y-1/2 bg-[#025253] hover:bg-[#025253]/90 h-8 w-8 sm:h-7 sm:w-7 touch-manipulation rounded-full"
              >
                <Send className="h-3 w-3 sm:h-3 sm:w-3" />
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
    
    <CitationsPanel
      sources={selectedSources}
      isOpen={isPanelOpen}
      onClose={() => setIsPanelOpen(false)}
    />
    </>
  );
};
