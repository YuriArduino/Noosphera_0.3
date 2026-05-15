"""
Prompt Management — Noosphera Nisaba.

Centralizes all system prompts, templates, and persona instructions.
Allows for decoupled evolution of the agent's personality and reasoning style.
"""

from typing import Optional
from nisaba.config.cognition import nisaba_cognition


class NisabaPrompts:
    """
    Collection of prompt templates for the Nisaba Agent.

    Design rationale:
        - Using a class-based structure allows for future localization or
          dynamic prompt swapping based on the agent's state.
        - Persona details are kept separate from formatting logic to ensure
          consistency across different nodes.
    """

    @classmethod
    def get_entity_extraction_prompt(
        cls,
        user_input: str,
        agent_response: str,
        session_id: str = "unknown",
    ) -> str:
        """
        Gera o prompt para extração estruturada de fatos persistíveis.
        """
        return (
            "Você extrai fatos persistíveis da conversa para alimentar um grafo.\n"
            "Retorne JSON válido. Não retorne Cypher. Não invente IDs.\n"
            "\n"
            "Formato obrigatório:\n"
            "{\n"
            '  "person": {"name": string | null, "age": number | null},\n'
            '  "preferences": [{"key": string, "value": string}],\n'
            '  "topics": [{"name": string}]\n'
            "}\n"
            "\n"
            "Regras:\n"
            "1. Extraia apenas fatos estáveis afirmados pelo usuário.\n"
            "2. Não transforme a resposta do assistente em fato.\n"
            "3. Se o usuário disser o próprio nome, preencha person.name.\n"
            "4. Se o usuário disser a própria idade, preencha person.age como número inteiro.\n"
            "5. Preferências devem usar chaves canônicas quando possível: favorite_animal, favorite_food, favorite_color.\n"
            "6. Para 'meu animal favorito é gato', use key favorite_animal e value gato.\n"
            "7. Tópicos são assuntos mencionados explicitamente, como Neo4j ou Cypher.\n"
            "8. Se não houver fatos claros, retorne exatamente: {\"person\": null, \"preferences\": [], \"topics\": []}\n"
            "9. Retorne somente JSON, sem Markdown.\n"
            "\n"
            "Exemplo:\n"
            '{"person": {"name": "Yuri", "age": 37}, "preferences": [{"key": "favorite_animal", "value": "gato"}], "topics": [{"name": "Neo4j"}]}\n'
            "\n"
            f"Session id: {session_id}\n"
            f"Conversa:\nUser: {user_input}\nAssistant: {agent_response}\n"
        )

    # ---------------------------------------------------------------------------
    # PERSONA DEFINITION
    # ---------------------------------------------------------------------------
    PERSONA = (
        "You are Nisaba, a specialized psychoanalytic assistant within the "
        "Noosphera ecosystem. Your goal is to provide deep, symbolic analysis "
        "and maintain a consistent, empathetic, yet professional therapeutic stance."
    )

    CORE_INSTRUCTIONS = (
        "1. Analyze the provided memory context before answering.\n"
        "2. Look for symbolic connections between the user's current query and past events.\n"
        "3. Use <thought> tags for your internal reasoning process.\n"
        "4. Be precise, grounded, and avoid generic AI platitudes."
    )

    # ---------------------------------------------------------------------------
    # TEMPLATE GENERATORS
    # ---------------------------------------------------------------------------

    @classmethod
    def get_main_system_prompt(cls, memory_context: Optional[str] = None) -> str:
        """
        Generates the primary system prompt for the conversation node.

        Args:
            memory_context: Optional string containing retrieved vector experiences.
        """
        prompt = f"{cls.PERSONA}\n\n" f"### Operational Instructions:\n{cls.CORE_INSTRUCTIONS}\n\n"

        if memory_context:
            prompt += (
                f"### Memory Context (SST):\n"
                f"The following records were retrieved from your long-term memory. "
                f"Use them to ensure continuity:\n{memory_context}\n"
            )

        return prompt

    @classmethod
    def get_reflection_prompt(cls, draft: str, memory_context: str) -> str:
        """
        Template for the reflection/revision node.
        Used to ensure the agent doesn't hallucinate ignorance when data is present.
        """
        return (
            "You are a response auditor. Review the draft below against the "
            "provided memory context. If the draft claims to 'not know' something "
            "that is clearly stated in the context, rewrite it for accuracy.\n\n"
            f"Context:\n{memory_context}\n\n"
            f"Draft to Revise:\n{draft}\n\n"
            "Output only the final corrected response."
        )
