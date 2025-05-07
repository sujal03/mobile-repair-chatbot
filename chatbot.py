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
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["FAQ_chatbot"]
embedding_collection = mongo_db["embeddings"]
chat_history_collection = mongo_db["chat_history"]

# Path to PDF source folder
PDF_SOURCE_FOLDER = "./pdf_source"

# Token limit for the embedding model
MAX_TOKENS = 8192
CHUNK_SIZE = 7000  # Conservative chunk size to account for token limits

def init_components(session_id: str):
    """Initialize necessary components for the chatbot session."""
    # Create database indexes if they don't exist
    try:
        # Create TTL index on chat history
        chat_history_collection.create_index(
            [("last_updated", pymongo.ASCENDING)], 
            expireAfterSeconds=60*60*24*30  # 30 days TTL
        )
        # Create session index
        chat_history_collection.create_index([("session_id", pymongo.ASCENDING)], unique=True)
        # Create embedding index
        embedding_collection.create_index([("file_name", pymongo.ASCENDING)])
        
        logger.info(f"Database indexes verified for session: {session_id}")
        
        # In a real application, we would initialize components like:
        # vectordb = create_or_load_vectordb()
        # retriever = vectordb.as_retriever()
        # llm = load_llm_model()
        
        # For compatibility with the current code structure:
        vectordb = "vectordb_placeholder"  # This would be an actual vectordb in production
        retriever = "retriever_placeholder"  # This would be an actual retriever in production
        llm = "llm_placeholder"  # This would be an actual LLM in production
        
        return (vectordb, retriever, llm)
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        # Return placeholders to avoid breaking the application
        return (None, None, None)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"  # Add newline between pages for better separation
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def chunk_text(text: str, max_tokens: int = CHUNK_SIZE) -> list:
    """Split text into chunks that respect the token limit while preserving paragraph and sentence boundaries."""
    if not text:
        return []

    # First split by paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_token_count = 0

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        
        for paragraph in paragraphs:
            paragraph_tokens = encoding.encode(paragraph)
            paragraph_token_count = len(paragraph_tokens)
            
            # If the paragraph is too large, split it by sentences
            if paragraph_token_count > max_tokens:
                sentences = paragraph.split(". ")
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    sentence_tokens = encoding.encode(sentence + "." if not sentence.endswith(".") else sentence)
                    sentence_token_count = len(sentence_tokens)
                    
                    # If adding this sentence would exceed the limit, start a new chunk
                    if current_token_count + sentence_token_count > max_tokens:
                        if current_chunk:
                            chunk_text = encoding.decode(current_chunk)
                            chunks.append(chunk_text)
                            current_chunk = sentence_tokens
                            current_token_count = sentence_token_count
                        else:
                            # This handles the case where a single sentence is too long
                            chunks.append(encoding.decode(sentence_tokens[:max_tokens]))
                            current_chunk = []
                            current_token_count = 0
                    else:
                        current_chunk.extend(sentence_tokens)
                        current_token_count += sentence_token_count
            else:
                # If adding this paragraph would exceed the limit, start a new chunk
                if current_token_count + paragraph_token_count > max_tokens:
                    if current_chunk:
                        chunk_text = encoding.decode(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = paragraph_tokens
                        current_token_count = paragraph_token_count
                else:
                    current_chunk.extend(paragraph_tokens)
                    current_token_count += paragraph_token_count
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = encoding.decode(current_chunk)
            chunks.append(chunk_text)
            
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        # Fallback to simple splitting if the token-based approach fails
        if not chunks and text:
            return [text[:4000]]  # Arbitrary character limit as a fallback
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
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def store_embeddings():
    """Process PDFs in the source folder and store embeddings in MongoDB."""
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_FOLDER, "*.pdf"))
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        
        # Check if we need to reprocess this file
        last_processed = embedding_collection.find_one({"file_name": file_name, "metadata.processed": True})
        file_stats = os.stat(pdf_path)
        
        # Skip if the file has been processed and hasn't been modified
        if last_processed and last_processed.get("metadata", {}).get("modified_time") == file_stats.st_mtime:
            logger.info(f"Embedding for {file_name} already exists and is up to date, skipping...")
            continue
        
        # If we're reprocessing, delete old embeddings
        if last_processed:
            embedding_collection.delete_many({"file_name": file_name})
            logger.info(f"Deleted old embeddings for {file_name} to reprocess")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {file_name}")
            continue

        text_chunks = chunk_text(text, max_tokens=CHUNK_SIZE)
        if not text_chunks:
            logger.warning(f"No valid chunks generated for {file_name}")
            continue

        processed_count = 0
        for i, chunk in enumerate(text_chunks):
            token_count = count_tokens(chunk)
            if token_count > MAX_TOKENS:
                logger.warning(f"Chunk {i+1} of {file_name} exceeds token limit ({token_count} tokens), truncating...")
                # Truncate the chunk instead of skipping
                chunk = chunk_text(chunk, max_tokens=MAX_TOKENS)[0]
                token_count = count_tokens(chunk)

            embedding = generate_embedding(chunk)
            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {i+1} of {file_name}")
                continue

            # Store with metadata
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
    """Calculate cosine similarity between two vectors."""
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

def search_relevant_embeddings(user_message: str, top_k: int = 3) -> list:
    """Search for relevant embeddings in MongoDB based on user message."""
    user_embedding = generate_embedding(user_message)
    if not user_embedding:
        logger.warning("Failed to generate embedding for user message")
        return []

    try:
        # Get all embeddings
        embeddings = list(embedding_collection.find({}, {"_id": 0, "embedding": 1, "text": 1, "file_name": 1, "chunk_index": 1}))
        similarities = []

        for doc in embeddings:
            stored_embedding = doc["embedding"]
            similarity = cosine_similarity(user_embedding, stored_embedding)
            similarities.append((similarity, doc["text"], doc["file_name"], doc.get("chunk_index", 0)))

        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        return []

def encode_image(image_file) -> str:
    """Encode image file to base64 string."""
    try:
        if isinstance(image_file, BytesIO):
            image_data = image_file.getvalue()
        else:
            image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return ""

def get_recent_conversation_summary(chat_history: list, max_messages: int = 4) -> str:
    """Generate a summary of recent conversation for context."""
    if not chat_history or len(chat_history) < 2:
        return ""
    
    recent_messages = chat_history[-max_messages*2:] if len(chat_history) > max_messages*2 else chat_history
    conversation_summary = "Recent conversation summary:\n"
    
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_summary += f"{role}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}\n"
    
    return conversation_summary

def generate_llm_response(user_message: str, vectordb, llm, chat_history: list, image_file=None) -> str:
    """Generate a response using the LLM with relevant embeddings and optional image."""
    try:
        # Search for relevant embeddings
        relevant_docs = search_relevant_embeddings(user_message)
        context = ""
        for similarity, text, file_name, chunk_index in relevant_docs:
            context += f"Relevant info from {file_name} (chunk {chunk_index}, similarity: {similarity:.2f}):\n{text[:500]}...\n\n"

        # Add conversation summary for better context awareness
        conversation_summary = get_recent_conversation_summary(chat_history)
        
        # Prepare system prompt
        system_prompt = f"""You are a mobile repair expert AI. Provide clear, practical solutions for mobile phone issues, including hardware and software problems. Offer step-by-step troubleshooting, repair advice, or recommendations for when to seek professional help. 

If an image is provided, analyze it for visible damage, error messages, or relevant details to assist with diagnostics. 

Keep responses concise, accurate, and user-friendly. Use the following relevant information if applicable:

{context}

{conversation_summary}

Important guidelines:
- Focus on accurate technical advice
- Provide step-by-step solutions when appropriate
- If you're unsure about specific details, be honest about limitations
- Prioritize safety (e.g., battery handling, water damage)
- Format complex instructions with bullet points or numbered steps
- Be concise but thorough
"""

        # Prepare messages with mobile repair prompt and context
        messages = [{
            "role": "system",
            "content": system_prompt
        }]

        # Add recent chat history (last 8 messages)
        for msg in chat_history[-8:]:
            role = msg["role"]
            if role == "bot":
                role = "assistant"
            
            # For image messages
            if msg.get("image_base64"):
                message_content = {"role": role, "content": []}
                message_content["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{msg['image_base64']}"}
                })
                message_content["content"].append({
                    "type": "text", 
                    "text": msg["message"]
                })
                messages.append(message_content)
            else:
                # For text-only messages
                messages.append({
                    "role": role,
                    "content": msg["message"]
                })

        # Prepare current user message
        if image_file:
            base64_image = encode_image(image_file)
            if base64_image:
                user_content = []
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
        else:
            # Text-only message
            messages.append({
                "role": "user", 
                "content": user_message
            })

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again or rephrase your question."

def generate_suggestions(user_message: str, chat_history: list = None) -> list:
    """Generate follow-up questions using OpenAI based on the user's message and conversation history."""
    try:
        # Include recent chat history for better context
        recent_context = ""
        if chat_history and len(chat_history) > 0:
            # Get last 3 exchanges (up to 6 messages)
            recent_messages = chat_history[-6:] if len(chat_history) >= 6 else chat_history
            recent_context = "Recent conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["message"][:100] + ("..." if len(msg["message"]) > 100 else "")
                recent_context += f"{role}: {content}\n"
        
        prompt = f"""
        You are a mobile repair expert AI. Based on the user's question and the conversation context, suggest 3 follow-up questions that the user might ask next to continue the conversation. The questions should be relevant to iPhone repair, troubleshooting, or mobile device maintenance. Keep the questions concise, practical, and directly related to the current conversation topic.

        {recent_context}
        
        User's latest message: "{user_message}"

        Provide the follow-up questions as a list of strings, like this:
        - "Question 1?"
        - "Question 2?"
        - "Question 3?"
        
        Important: Make sure the suggestions are directly relevant to what was just discussed.
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
        suggestions = [s.strip('- ').strip('"').strip() for s in suggestions if s.strip()]
        
        # Ensure we have exactly 3 suggestions
        while len(suggestions) < 3:
            suggestions.append("What other mobile repair services do you offer?")
        
        return suggestions[:3]
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return [
            "How can I fix a cracked screen?",
            "What are common troubleshooting steps for battery issues?",
            "How do I back up my device data before repair?"
        ]

def get_chat_history(session_id: str) -> list:
    """Retrieve chat history for a specific session."""
    try:
        doc = chat_history_collection.find_one({"session_id": session_id})
        if doc:
            return doc.get("messages", [])
        return []
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return []

def save_chat_message(session_id: str, message: dict) -> bool:
    """Save a message to the chat history."""
    try:
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow()
            
        # Update or create the document
        result = chat_history_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message},
                "$set": {"last_updated": datetime.utcnow()}
            },
            upsert=True
        )
        
        return result.acknowledged
    except Exception as e:
        logger.error(f"Error saving chat message: {str(e)}")
        return False

# if __name__ == "__main__":
#     store_embeddings()