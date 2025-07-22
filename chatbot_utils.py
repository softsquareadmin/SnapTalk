import os
from openai import OpenAI, OpenAIError
from pinecone import Pinecone
import dotenv

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

def embed_query(text, model="text-embedding-3-small"):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return None

def search_pinecone(query_embedding, top_k=5):
    try:
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return response.matches
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

def build_context(matches):
    return "\n\n".join([m.metadata.get("text", "") for m in matches if m.metadata and "text" in m.metadata])

def generate_gpt_reply(chat_history, context, user_input):
    system_prompt = (
        "You are a helpful assistant. "
        "Use the provided context, which is extracted from pamphlet images, to answer questions. "
        "If the context does not contain enough information, say you don't know. "
        "Respond in clear, professional language. "
        "Use markdown formatting where appropriate:\n"
        "- Bold important words\n"
        "- Use bullet points for lists\n"
        "- Keep answers concise and uniform\n"
        "- Add line breaks between paragraphs for readability\n"
        "Only answer using the information from the pamphlet context provided."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "Sorry, there was an error generating a response." 