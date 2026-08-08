"""Gemini chat and embedding model factories."""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings


def get_chat(settings: Settings, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def get_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embed_model,
        google_api_key=settings.google_api_key,
        # gemini-embedding-001 defaults to 3072 dims; truncate to match the
        # Pinecone index. Cosine similarity is unaffected by the truncation
        # scale (Pinecone normalizes internally).
        output_dimensionality=settings.embed_dim,
    )
