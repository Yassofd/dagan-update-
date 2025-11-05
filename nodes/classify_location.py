"""
Node CLASSIFY_LOCATION - Détermine si l'utilisateur est resident ou diaspora
"""

import os
from typing import Dict, Literal
from openai import OpenAI

def classify_location(state: Dict) -> Dict:
    """
    Classifie la question pour déterminer si l'utilisateur est resident (Togo) ou diaspora (étranger).
    
    Args:
        state (Dict): État contenant les messages
    
    Returns:
        Dict avec clé "user_location" ("resident" ou "diaspora")
    """
    
    # Extraire la dernière question utilisateur
    messages = state.get("messages", [])
    if not messages:
        return {"user_location": "resident"}
    
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    print(f"📍 Classification localisation: '{question[:50]}...'")
    
    # Configuration LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Prompt de classification
    classification_prompt = f"""Tu es un classificateur de contexte géographique pour Dagan.

Détermine si l'utilisateur est un "resident" (habite au Togo) ou "diaspora" (habite à l'étranger).

**RESIDENT** (réponds "resident") - Indices:
- Aucune mention d'un pays ou lieu spécifique
- Mention explicite du Togo, Lomé, ou villes togolaises
- Contexte suggérant une présence physique au Togo
- Questions par défaut sans contexte géographique

**DIASPORA** (réponds "diaspora") - Indices:
- Mention explicite d'un pays étranger (France, Belgique, Canada, Allemagne, etc.)
- Mention d'une ville/région étrangère (Paris, Bruxelles, etc.)
- Phrases comme "étant en...", "depuis la...", "de l'étranger", "abroad"
- Mention d'une situation d'expatriation ou d'immigration

Question: "{question}"

Réponds UNIQUEMENT par "resident" ou "diaspora"."""
    
    try:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if "diaspora" in result:
            print("🌍 Classifié: DIASPORA")
            return {"user_location": "diaspora"}
        else:
            print("🏠 Classifié: RESIDENT")
            return {"user_location": "resident"}
    
    except Exception as e:
        print(f"⚠️ Erreur classification, défaut vers resident: {e}")
        return {"user_location": "resident"}
