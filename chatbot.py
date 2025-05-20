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
import logging
from datetime import datetime
from prompts import main_prompt, get_follow_up_questions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB setup
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["FAQ_chatbot"]
embedding_collection = mongo_db["embeddings"]
chat_history_collection = mongo_db["chat_history"]

# Create database indexes (runs once on module import)
try:
    chat_history_collection.create_index(
        [("last_updated", pymongo.ASCENDING)], expireAfterSeconds=60 * 60 * 24 * 30  # 30 days TTL
    )
    chat_history_collection.create_index([("session_id", pymongo.ASCENDING)], unique=True)
    embedding_collection.create_index([("file_name", pymongo.ASCENDING)])
    logger.info("Database indexes created successfully")
except Exception as e:
    logger.error(f"Error creating database indexes: {str(e)}")

# Constants
PDF_SOURCE_FOLDER = "./pdf_source"
MAX_TOKENS = 8192
CHUNK_SIZE = 7000


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text (str): Text to tokenize.

    Returns:
        int: Number of tokens.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE) -> list:
    """Split text into chunks respecting token limits and paragraph boundaries.

    Args:
        text (str): Text to chunk.
        max_tokens (int): Max tokens per chunk.

    Returns:
        list: List of text chunks.
    """
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_token_count = 0
    encoding = tiktoken.get_encoding("cl100k_base")

    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        paragraph_token_count = len(paragraph_tokens)
        if paragraph_token_count > max_tokens:
            sentences = paragraph.split(". ")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence_tokens = encoding.encode(sentence + "." if not sentence.endswith(".") else sentence)
                if current_token_count + len(sentence_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(encoding.decode(current_chunk))
                        current_chunk = sentence_tokens
                        current_token_count = len(sentence_tokens)
                    else:
                        chunks.append(encoding.decode(sentence_tokens[:max_tokens]))
                        current_chunk = []
                        current_token_count = 0
                else:
                    current_chunk.extend(sentence_tokens)
                    current_token_count += len(sentence_tokens)
        else:
            if current_token_count + paragraph_token_count > max_tokens:
                if current_chunk:
                    chunks.append(encoding.decode(current_chunk))
                    current_chunk = paragraph_tokens
                    current_token_count = paragraph_token_count
            else:
                current_chunk.extend(paragraph_tokens)
                current_token_count += paragraph_token_count

    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    return chunks


def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI's embedding model.

    Args:
        text (str): Text to embed.

    Returns:
        list: Embedding vector.
    """
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []


def store_embeddings():
    """Process PDFs and store embeddings in MongoDB (run separately)."""
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_FOLDER, "*.pdf"))
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        last_processed = embedding_collection.find_one({"file_name": file_name, "metadata.processed": True})
        file_stats = os.stat(pdf_path)

        if last_processed and last_processed.get("metadata", {}).get("modified_time") == file_stats.st_mtime:
            logger.info(f"Embedding for {file_name} up to date, skipping...")
            continue

        if last_processed:
            embedding_collection.delete_many({"file_name": file_name})
            logger.info(f"Deleted old embeddings for {file_name}")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {file_name}")
            continue

        text_chunks = chunk_text(text)
        processed_count = 0
        for i, chunk in enumerate(text_chunks):
            token_count = count_tokens(chunk)
            if token_count > MAX_TOKENS:
                chunk = chunk_text(chunk, max_tokens=MAX_TOKENS)[0]
            embedding = generate_embedding(chunk)
            if embedding:
                embedding_collection.insert_one({
                    "file_name": file_name,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "processed": True,
                        "modified_time": file_stats.st_mtime,
                        "token_count": token_count,
                        "processed_at": datetime.utcnow()
                    }
                })
                processed_count += 1
        logger.info(f"Stored {processed_count} embeddings for {file_name}")


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.

    Returns:
        float: Similarity score.
    """
    try:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0


def search_relevant_embeddings(user_message: str, top_k: int = 3) -> list:
    """Search for relevant embeddings in MongoDB.

    Args:
        user_message (str): User's message.
        top_k (int): Number of top results.

    Returns:
        list: List of (similarity, text, file_name, chunk_index) tuples.
    """
    user_embedding = generate_embedding(user_message)
    if not user_embedding:
        logger.warning("Failed to generate user message embedding")
        return []

    try:
        embeddings = list(embedding_collection.find({}, {"_id": 0, "embedding": 1, "text": 1, "file_name": 1, "chunk_index": 1}))
        similarities = [
            (cosine_similarity(user_embedding, doc["embedding"]), doc["text"], doc["file_name"], doc.get("chunk_index", 0))
            for doc in embeddings
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        return []


def encode_image(image_file) -> str:
    """Encode image to base64 string.

    Args:
        image_file: Image file object or BytesIO.

    Returns:
        str: Base64-encoded image.
    """
    try:
        image_data = image_file.getvalue() if isinstance(image_file, BytesIO) else image_file.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return ""


def get_recent_conversation_summary(chat_history: list, max_messages: int = 4) -> str:
    """Summarize recent conversation for context.

    Args:
        chat_history (list): List of chat messages.
        max_messages (int): Max recent messages to include.

    Returns:
        str: Conversation summary.
    """
    if not chat_history or len(chat_history) < 2:
        return ""
    recent = chat_history[-max_messages * 2:] if len(chat_history) > max_messages * 2 else chat_history
    summary = "Recent conversation summary:\n"
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        summary += f"{role}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}\n"
    return summary


def generate_llm_response(user_message: str, chat_history: list, image_file=None) -> str:
    """Generate an LLM response with context and optional image.

    Args:
        user_message (str): User's message.
        chat_history (list): Previous chat messages.
        image_file: Optional image file.

    Returns:
        str: LLM-generated response.
    """
    try:
        # Gather context from embeddings
        relevant_docs = search_relevant_embeddings(user_message)
        context = "".join(
            f"Relevant info from {file} (chunk {idx}, similarity: {sim:.2f}):\n{text[:500]}...\n\n"
            for sim, text, file, idx in relevant_docs
        )

        # Add conversation summary
        conversation_summary = get_recent_conversation_summary(chat_history)

        # System prompt
        system_prompt = main_prompt(context, conversation_summary)

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-8:]:
            role = "assistant" if msg["role"] == "bot" else msg["role"]
            if msg.get("image_base64"):
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg['image_base64']}"}},
                        {"type": "text", "text": msg["message"]}
                    ]
                })
            else:
                messages.append({"role": role, "content": msg["message"]})

        # Add current user message
        if image_file:
            base64_image = encode_image(image_file)
            user_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
            user_content.append({
                "type": "text",
                "text": user_message or "Analyze the image for mobile phone issues."
            })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_message})

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=500, temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error. Please try again or rephrase your question."


def generate_suggestions(user_message: str, chat_history: list = None) -> list:
    """Generate follow-up questions based on user message and history.

    Args:
        user_message (str): Latest user message.
        chat_history (list): Previous chat messages.

    Returns:
        list: Three suggested questions.
    """
    try:
        recent_context = ""
        if chat_history:
            recent = chat_history[-6:] if len(chat_history) >= 6 else chat_history
            recent_context = "Recent conversation:\n" + "".join(
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}\n"
                for msg in recent
            )

        prompt = get_follow_up_questions(recent_context, user_message)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Mobile repair expert AI"}, {"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.5
        )
        suggestions = [s.strip('- ').strip('"') for s in response.choices[0].message.content.strip().split('\n') if s.strip()]
        while len(suggestions) < 3:
            suggestions.append("What other repair services do you offer?")
        return suggestions[:3]
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return [
            "How can I fix a cracked screen?",
            "What are battery troubleshooting steps?",
            "How do I back up my device?"
        ]