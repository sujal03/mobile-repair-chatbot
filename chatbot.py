from openai import OpenAI
import os
from dotenv import load_dotenv
import pymongo
from PyPDF2 import PdfReader
import glob
import numpy as np
import tiktoken
import base64
from io import BytesIO

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["FAQ_chatbot"]
embedding_collection = mongo_db["embeddings"]

# Path to PDF source folder
PDF_SOURCE_FOLDER = "./pdf_source"

# Token limit for the embedding model
MAX_TOKENS = 8192
CHUNK_SIZE = 7000  # Conservative chunk size to account for token limits

def init_components(session_id: str):
    return (None, None, None)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

def chunk_text(text: str, max_tokens: int = CHUNK_SIZE) -> list:
    """Split text into chunks that respect the token limit."""
    if not text:
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for token in tokens:
        current_chunk.append(token)
        current_token_count += 1
        if current_token_count >= max_tokens:
            chunk_text = encoding.decode(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_token_count = 0

    if current_chunk:
        chunk_text = encoding.decode(current_chunk)
        chunks.append(chunk_text)

    return chunks

def generate_embedding(text: str) -> list:
    """Generate embedding for the given text using OpenAI's embedding model."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []

def store_embeddings():
    """Process PDFs in the source folder and store embeddings in MongoDB."""
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_FOLDER, "*.pdf"))
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)

        if embedding_collection.find_one({"file_name": file_name}):
            print(f"Embedding for {file_name} already exists, skipping...")
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from {file_name}")
            continue

        text_chunks = chunk_text(text, max_tokens=CHUNK_SIZE)
        if not text_chunks:
            print(f"No valid chunks generated for {file_name}")
            continue

        for i, chunk in enumerate(text_chunks):
            token_count = count_tokens(chunk)
            if token_count > MAX_TOKENS:
                print(f"Chunk {i+1} of {file_name} exceeds token limit ({token_count} tokens), skipping...")
                continue

            embedding = generate_embedding(chunk)
            if not embedding:
                print(f"Failed to generate embedding for chunk {i+1} of {file_name}")
                continue

            embedding_collection.insert_one({
                "file_name": file_name,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding
            })
            print(f"Stored embedding for chunk {i+1} of {file_name}")

def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def search_relevant_embeddings(user_message: str, top_k: int = 3) -> list:
    """Search for relevant embeddings in MongoDB based on user message."""
    user_embedding = generate_embedding(user_message)
    if not user_embedding:
        return []

    embeddings = embedding_collection.find()
    similarities = []

    for doc in embeddings:
        stored_embedding = doc["embedding"]
        similarity = cosine_similarity(user_embedding, stored_embedding)
        similarities.append((similarity, doc["text"], doc["file_name"], doc.get("chunk_index", 0)))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def encode_image(image_file) -> str:
    """Encode image file to base64 string."""
    try:
        if isinstance(image_file, BytesIO):
            image_data = image_file.getvalue()
        else:
            image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return ""

def generate_llm_response(user_message: str, vectordb, llm, chat_history: list, image_file=None) -> str:
    """Generate a response using the LLM with relevant embeddings and optional image."""
    try:
        # Search for relevant embeddings
        relevant_docs = search_relevant_embeddings(user_message)
        context = ""
        for similarity, text, file_name, chunk_index in relevant_docs:
            context += f"Relevant info from {file_name} (chunk {chunk_index}, similarity: {similarity:.2f}):\n{text[:500]}...\n\n"

        # Prepare messages with mobile repair prompt and context
        messages = [{
            "role": "system",
            "content": """You are a mobile repair expert AI. Provide clear, practical solutions for mobile phone issues, including hardware and software problems. Offer step-by-step troubleshooting, repair advice, or recommendations for when to seek professional help. If an image is provided, analyze it for visible damage, error messages, or relevant details to assist with diagnostics. Keep responses concise, accurate, and user-friendly. Use the following relevant information if applicable:\n\n""" + context
        }]

        # Add recent chat history (last 8 messages)
        for msg in chat_history[-8:]:
            role = msg["role"]
            if role == "bot":
                role = "assistant"
            message_content = {"role": role, "content": []}
            if msg.get("image_base64"):
                message_content["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{msg['image_base64']}"}
                })
            message_content["content"].append({
                "type": "text",
                "text": msg["message"]
            })
            messages.append(message_content)

        # Prepare current user message
        user_content = []
        if image_file:
            base64_image = encode_image(image_file)
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        if user_message:
            user_content.append({
                "type": "text",
                "text": user_message
            })
        else:
            user_content.append({
                "type": "text",
                "text": "Please analyze the provided image for any mobile phone issues."
            })

        messages.append({"role": "user", "content": user_content})

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def generate_suggestions(user_message: str) -> list:
    """Generate follow-up questions using OpenAI based on the user's message."""
    try:
        prompt = f"""
        You are a mobile repair expert AI. Based on the user's question or statement, suggest 3 follow-up questions that the user might ask next to continue the conversation. The questions should be relevant to iPhone repair, troubleshooting, or maintenance. Keep the questions concise and practical.

        User's message: "{user_message}"

        Provide the follow-up questions as a list of strings, like this:
        - "Question 1?"
        - "Question 2?"
        - "Question 3?"
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mobile repair expert AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        suggestions = response.choices[0].message.content.strip().split('\n')
        # Clean up suggestions (remove bullet points and trim)
        suggestions = [s.strip('- ').strip() for s in suggestions if s.strip()]
        # Ensure we have exactly 3 suggestions
        while len(suggestions) < 3:
            suggestions.append("What are common troubleshooting steps for this issue?")
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        return [
            "Can you help with another device issue?",
            "What are common troubleshooting steps for devices?",
            "How do I back up my device data?"
        ]

if __name__ == "__main__":
    store_embeddings()