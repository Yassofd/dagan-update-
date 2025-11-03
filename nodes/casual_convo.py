"""
Node CASUAL_CONVO - Gestion des conversations informelles
"""

import os
from typing import Dict
from openai import OpenAI
from langchain_core.messages import AIMessage

def casual_convo(state: Dict) -> Dict:
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="Bonjour ! Je suis Dagan, votre assistant. Comment puis-je vous aider ?")]}

    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)

    print(f"Casual conversation: '{question[:50]}...'")
    print(f"Historique: {len(messages)} messages")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    conversation_history = []
    for msg in messages[:-1]:
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        conversation_history.append({"role": role, "content": content})

    casual_system_prompt = "Tu es Dagan, un assistant IA amical et sympathique specialise dans l'aide administrative togolaise. Tu dois te souvenir des informations que l'utilisateur t'a donnees precedemment dans la conversation (comme son nom, sa situation, ses etudes, etc.)"

    casual_user_prompt = f"L'utilisateur te pose : {question}\n\nReponds en restant amical et en tenant compte de tout l'historique de la conversation."

    try:
        llm_messages = [
            {"role": "system", "content": casual_system_prompt},
            *conversation_history,
            {"role": "user", "content": casual_user_prompt}
        ]
        
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0.7,  
            max_tokens=200,
            messages=llm_messages
        )

        answer = response.choices[0].message.content.strip()
        print(f"Casual response: '{answer[:50]}...'")

        new_messages = messages + [AIMessage(content=answer)]
        return {"messages": new_messages}

    except Exception as e:
        print(f"Erreur casual response: {e}")
        fallback = "Desole, je n'ai pas bien compris. Je suis Dagan, votre assistant pour les demarches administratives au Togo. Comment puis-je vous aider ?"
        return {"messages": messages + [AIMessage(content=fallback)]}
