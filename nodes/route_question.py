"""
Node ROUTE_QUESTION - Routeur intelligent entre conversations casual et questions administratives
avec conscience du contexte conversationnel
"""

import os
from typing import Dict, Literal
from openai import OpenAI
from langchain_core.messages import HumanMessage as LangchainHumanMessage, AIMessage as LangchainAIMessage

def route_question(state: Dict) -> Dict:
    """
    Route la question vers casual_convo ou agent_rag selon le type de question.
    
    Prend en compte l'historique conversationnel:
    - Si les messages précédents étaient admin, les suivi restent admin
    - Analyse le contexte global, pas juste la dernière question
    
    Args:
        state (Dict): État contenant les messages

    Returns:
        Dict avec clé "question_type" ("casual" ou "admin")
    """

    # Extraire tous les messages
    messages = state.get("messages", [])
    if not messages:
        return {"question_type": "casual"}

    # Extraire les messages utilisateur et assistant
    user_messages = [msg for msg in messages if isinstance(msg, LangchainHumanMessage)]
    assistant_messages = [msg for msg in messages if isinstance(msg, LangchainAIMessage)]
    
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)

    print(f"🔀 Routing question: '{question[:50]}...'")
    print(f"   Historique: {len(user_messages)} messages utilisateur, {len(assistant_messages)} messages assistant")

    # Configuration LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # LOGIQUE 1: Si c'est une question de suivi (historique > 1 message utilisateur)
    # et que la réponse précédente était administrative, rester en admin
    if len(user_messages) > 1 and len(assistant_messages) > 0:
        last_assistant_msg = assistant_messages[-1]
        last_assistant_content = last_assistant_msg.content if hasattr(last_assistant_msg, 'content') else str(last_assistant_msg)
        
        # Détecter si la réponse précédente contenait du contenu administratif
        admin_keywords = [
            "procédure", "document", "administrative", "pièces", "formulaire",
            "coût", "délai", "demande", "passeport", "identité", "carte",
            "ministère", "service public", "conditions", "étapes", "gouv.tg",
            "diplôme", "permis", "licence", "attestation", "certificat",
            "acte", "contrat", "immatriculation", "enregistrement", "taxe"
        ]
        
        content_lower = last_assistant_content.lower()
        admin_score = sum(1 for keyword in admin_keywords if keyword in content_lower)
        
        # Si au moins 2 mots-clés admin trouvés, c'est une conversation admin
        if admin_score >= 2:
            print("✓ Conversation précédente = ADMIN, maintien en ADMIN pour suivi")
            return {"question_type": "admin"}

    # Prompt de classification
    routing_prompt = f"""Tu es un routeur intelligent pour Dagan, assistant togolais spécialisé dans les procédures administratives.

Classifie cette question en "casual" ou "admin" :

**CASUAL** (réponds "casual") - Conversations informelles :
- Salutations : "bonjour", "salut", "ça va ?", "comment allez-vous ?"
- Questions générales : météo, actualités, sport, divertissement
- Conversation personnelle : "tu es qui ?", "que fais-tu ?", "parle-moi de toi"
- Questions fermées simples : "oui", "non", "peut-être", réponses courtes
- Questions de politesse : "merci", "au revoir", "à bientôt"
- Questions vagues sans contexte administratif : "et pour..." (si vraiment flou)

**ADMIN** (réponds "admin") - Questions administratives togolaises :
- Documents officiels : passeport, carte d'identité, acte de naissance
- Éducation : inscription scolaire, bourses, diplômes
- Emploi : recherche d'emploi, sécurité sociale, retraite
- Santé : assurance maladie, soins médicaux
- Fiscalité : impôts, taxes, déclarations
- Entreprises : création société, permis d'exploitation
- Logement : permis construire, propriété foncière
- Transport : permis conduire, immatriculation véhicule
- Justice : procédures judiciaires, tribunaux
- Télécommunications : abonnement internet, téléphone
- Agriculture : subventions, certifications
- Sécurité : police, gendarmerie, protection civile
- Questions de suivi sur des procédures : "et pour...", "comment si...", "et pour les conditions..."

Question : "{question}"

Réponds UNIQUEMENT par "casual" ou "admin"."""

    try:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[{"role": "user", "content": routing_prompt}]
        )

        result = response.choices[0].message.content.strip().lower()

        if "casual" in result:
            print("🎯 Routed to: CASUAL_CONVO")
            return {"question_type": "casual"}
        else:
            print("🎯 Routed to: AGENT_RAG")
            return {"question_type": "admin"}

    except Exception as e:
        print(f"⚠️ Erreur routing, défaut vers admin: {e}")
        return {"question_type": "admin"}