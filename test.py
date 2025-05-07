# main.py
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, session
from chatbot import (
    init_components, 
    generate_llm_response, 
    generate_suggestions, 
    get_chat_history,
    save_chat_message
)
import os
import pymongo
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta
import logging
from io import BytesIO
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit uploads to 10MB

# MongoDB setup

mongo_client = pymongo.MongoClient(
    os.getenv("MONGO_URI"), 
    serverSelectionTimeoutMS=5000  # 5 second timeout
)
# Test the connection
mongo_client.server_info()
mongo_db = mongo_client["FAQ_chatbot"]
chat_history_collection = mongo_db["chat_history"]
logger.info("MongoDB connection successful")


# In-memory fallback storage
memory_chat_history = {}

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def get_or_create_session():
    """Get existing session or create a new one."""
    session_id = request.cookies.get('session_id')
    
    # If no cookie or invalid session, create new session
    if not session_id or not is_valid_session(session_id):
        session_id = generate_session_id()
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Using existing session: {session_id}")
        
    return session_id

def is_valid_session(session_id):
    """Check if a session exists and is valid."""
    try:
        if chat_history_collection:
            # Check MongoDB
            session = chat_history_collection.find_one({"session_id": session_id})
            return session is not None
        else:
            # Check in-memory storage
            return session_id in memory_chat_history
    except Exception as e:
        logger.error(f"Error checking session validity: {str(e)}")
        return False
    
@app.route('/', methods=['GET'])
def index():
    """Redirect to chatbot interface."""
    return redirect(url_for('chatbot'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        # Get or create a session ID
        session_id = get_or_create_session()
        
        # Get the chat history
        history_doc = chat_history_collection.find_one({"session_id": session_id})
        chat_history = history_doc.get("messages", []) if history_doc else []
        
        # Render the template with chat history
        response = make_response(render_template('chatbot.html', chat_history=chat_history))
        
        # Set session cookie (30 days expiry)
        response.set_cookie('session_id', session_id, max_age=86400 * 30)
        
        return response
    
    elif request.method == 'POST':
        # Handle API request (AJAX)
        try:
            # Get session ID from cookie
            session_id = request.cookies.get('session_id')
            if not session_id:
                session_id = generate_session_id()
                logger.info(f"New session created in POST: {session_id}")
            
            # Initialize components
            vectordb, retriever, llm = init_components(session_id)
            
            # Get user input
            user_message = request.form.get('user_message', '').strip()
            image_file = request.files.get('image')
            
            # Validate input
            if not user_message and not image_file:
                return jsonify({"error": "Please provide a message or an image."}), 400
            
            # Validate image file if provided
            if image_file:
                if not image_file.content_type.startswith('image/'):
                    return jsonify({"error": "Invalid file type. Please upload an image."}), 400
                
                # 5MB limit
                if image_file.content_length > 5 * 1024 * 1024:
                    return jsonify({"error": "Image size exceeds 5MB limit."}), 400
            
            
            history_doc = chat_history_collection.find_one({"session_id": session_id})
            messages = history_doc.get("messages", []) if history_doc else []
            
            # Create user message entry
            user_message_entry = {
                "role": "user",
                "message": user_message if user_message else "Image uploaded for analysis",
                "timestamp": datetime.utcnow()
            }
            
            # Handle image storage
            if image_file:
                import base64
                image_data = BytesIO(image_file.read())
                image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
                user_message_entry["image_base64"] = image_base64
                user_message_entry["image_url"] = f"data:image/jpeg;base64,{image_base64}"
            
            # Add user message to history
            messages.append(user_message_entry)
            
            chat_history_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": user_message_entry},
                    "$set": {"last_updated": datetime.utcnow()}
                },
                upsert=True
            )
            
            logger.info(f"User message saved for session {session_id}")
            
            # Generate bot response (with timeout handling)
            start_time = time.time()
            bot_response = generate_llm_response(
                user_message,
                vectordb,
                llm,
                messages,
                image_file=BytesIO(image_data.getvalue()) if image_file else None
            )
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            
            # Create bot message entry
            bot_message_entry = {
                "role": "bot",
                "message": bot_response,
                "timestamp": datetime.utcnow()
            }
            
            # Add bot message to history
            messages.append(bot_message_entry)
            
            
            chat_history_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": bot_message_entry},
                    "$set": {"last_updated": datetime.utcnow()}
                }
            )
                
            logger.info(f"Bot response saved for session {session_id}")
            
            # Generate new suggestions based on the conversation
            suggestions = generate_suggestions(user_message, messages)
            
            # Create the response
            response_data = {
                "bot_response": bot_response,
                "user_message": user_message,
                "timestamp": bot_message_entry["timestamp"].isoformat(),
                "suggestions": suggestions
            }
            
            response = jsonify(response_data)
            
            # Set/refresh session cookie
            response.set_cookie('session_id', session_id, max_age=86400 * 30)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return jsonify({
                "error": "An error occurred while processing your request.",
                "details": str(e) if app.debug else "Please try again later."
            }), 500

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    """Generate and return follow-up question suggestions."""
    try:
        # Get the request data
        data = request.get_json()
        user_message = data.get('user_message', '').strip()
        session_id = request.cookies.get('session_id')
        
        if not user_message:
            return jsonify({"error": "No user message provided."}), 400
        
        # Get chat history for context-aware suggestions
        if chat_history_collection and session_id:
            history_doc = chat_history_collection.find_one({"session_id": session_id})
            messages = history_doc.get("messages", []) if history_doc else []
        else:
            messages = memory_chat_history.get(session_id, []) if session_id else []
        
        # Generate suggestions
        suggestions = generate_suggestions(user_message, messages)
        
        return jsonify({"suggestions": suggestions})
    
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return jsonify({
            "error": "Failed to generate suggestions.",
            "suggestions": [
                "How can I fix my screen?",
                "What should I do about battery drain?",
                "How do I reset my device?",
            ]
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=7000, host='0.0.0.0')


# chatbot.py
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


# chatbot.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aishwaray - AI Chatbot for Electronics Refurbishing</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.0.12/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@2.3.6/dist/purify.min.js"></script>
    <style>
        :root {
            --primary: #3A45D8;
            --primary-light: #E8EAFF;
            --secondary: #8D93C9;
            --text-primary: #1C1D21;
            --text-secondary: #6D717A;
            --light-bg: #F5F8FF;
            --white: #FFFFFF;
            --border-color: #E1E5EE;
            --gradient-start: #F0F4FF;
            --gradient-end: #FFE1E4;
            --user-msg-bg: #F0F4FF;
            --bot-msg-bg: #FFFFFF;
            --bot-icon-bg: #3A45D8;
            --box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .app-container {
            display: flex;
            width: 100%;
            height: 100vh;
        }

        .sidebar {
            width: 380px;
            background: var(--light-bg);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            position: fixed;
            height: 100%;
            z-index: 1000;
            box-shadow: var(--box-shadow);
            overflow-y: auto;
        }

        .sidebar-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
        }

        .suggestions {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .suggestions-mobile {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            padding: 1.5rem;
            background: var(--white);
            border-top: 1px solid var(--border-color);
        }

        .suggestion-chip {
            background: var(--white);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 0.95rem;
            font-weight: 500;
            box-shadow: var(--box-shadow);
        }

        .suggestions-mobile .suggestion-chip {
            background: var(--white);
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            border-radius: 10px;
        }

        .suggestion-chip:hover {
            background: var(--primary-light);
            border-color: var(--primary);
            transform: translateY(-2px);
            color: var(--primary);
        }

        .main-content {
            flex: 1;
            margin-left: 380px;
            display: flex;
            flex-direction: column;
            position: relative;
            background: var(--white);
        }

        .chat-header {
            padding: 1.5rem 2rem;
            background: var(--white);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
        }

        .chat-header-logo {
            width: 40px;
            height: 40px;
            background: #000;
            border-radius: 10px;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .chat-header-logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .chat-header h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
            color: var(--text-primary);
        }

        .chat-messages {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            background: var(--white);
            -webkit-overflow-scrolling: touch;
        }

        .message {
            max-width: 80%;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
            font-size: 0.95rem;
        }

        .user-message {
            background: var(--user-msg-bg);
            margin-left: auto;
            color: var(--text-primary);
            align-self: flex-end;
        }

        .bot-message {
            background: var(--bot-msg-bg);
            margin-right: auto;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            align-self: flex-start;
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }

        .message-content {
            word-wrap: break-word;
        }

        .message-timestamp {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            text-align: right;
        }

        .bot-icon {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--bot-icon-bg);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--white);
            font-size: 1rem;
            flex-shrink: 0;
        }

        .input-container {
            padding: 1.5rem 2rem;
            background: var(--white);
            border-top: 1px solid var(--border-color);
            position: sticky;
            bottom: 0;
        }

        .input-group {
            background: var(--white);
            border-radius: 12px;
            padding: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid var(--border-color);
        }

        .form-control {
            border: none;
            background: transparent;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            flex: 1;
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
        }

        .form-control:focus {
            box-shadow: none;
            outline: none;
        }

        .form-control::placeholder {
            color: var(--text-secondary);
        }

        .send-btn, .image-upload-label {
            background: var(--primary);
            color: var(--white);
            border: none;
            border-radius: 10px;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .image-upload-label {
            background: var(--white);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .send-btn:hover {
            background: #2a33b0;
        }

        .image-upload-label:hover {
            background: var(--light-bg);
        }

        .image-preview {
            max-width: 80px;
            max-height: 80px;
            object-fit: cover;
            border-radius: 8px;
            margin: 0.5rem;
            display: none;
        }

        .loading-dots {
            display: inline-flex;
            gap: 0.3rem;
        }

        .loading-dots span {
            width: 6px;
            height: 6px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: bounce 1.2s infinite ease-in-out;
        }

        .loading-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .loading-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }

        .message-image {
            max-width: 100%;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .message-text p {
            margin-bottom: 0.75rem;
        }

        .message-text p:last-child {
            margin-bottom: 0;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-6px); }
        }

        @media (max-width: 992px) {
            .sidebar {
                display: none;
            }

            .main-content {
                margin-left: 0;
            }

            .suggestions-mobile {
                display: flex;
            }
        }

        @media (min-width: 993px) {
            .suggestions-mobile {
                display: none;
            }
        }

        @media (max-width: 768px) {
            .message {
                max-width: 85%;
            }
        }

        @media (max-width: 576px) {
            .chat-messages, .input-container {
                padding: 1rem;
            }

            .message {
                max-width: 90%;
                padding: 0.875rem 1rem;
            }

            .chat-header {
                padding: 1rem;
            }

            .chat-header h2 {
                font-size: 1.1rem;
            }

            .suggestions-mobile {
                padding: 1rem;
                gap: 0.5rem;
            }

            .suggestions-mobile .suggestion-chip {
                padding: 0.6rem 0.875rem;
                font-size: 0.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">Popular Question</div>
            <div id="suggestions-desktop" class="suggestions">
                <div class="suggestion-chip">What is the cost of screen replacement?</div>
                <div class="suggestion-chip">Want to compare two devices?</div>
                <div class="suggestion-chip">Do you repair water-damaged phones?</div>
                <div class="suggestion-chip">What's your return policy?</div>
                <div class="suggestion-chip">Do you offer screen replacement services?</div>
            </div>
        </div>

        <div class="main-content">
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="flash-messages position-fixed top-0 end-0 p-3" style="z-index: 1050; max-width: 400px">
                {% for category, message in messages %}
                <div class="alert alert-{{ category }}">
                    <i class="fas fa-{% if category == 'success' %}check-circle{% else %}exclamation-circle{% endif %} me-2"></i>
                    {{ message }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endwith %}

            <div class="chat-header">
                <div class="chat-header-logo">
                    <img src="/api/placeholder/40/40" alt="Aishwaray Logo">
                </div>
                <h2>Aishwaray - Chatbot</h2>
            </div>

            <div class="chat-messages" id="chat-messages">
                {% for msg in chat_history %}
                {% if msg.role == 'user' %}
                <div class="message user-message">
                    {% if msg.image_url %}
                    <img src="{{ msg.image_url }}" alt="User uploaded image" class="message-image">
                    {% endif %}
                    <div class="message-content">
                        <div class="message-text">{{ msg.message | safe }}</div>
                    </div>
                    <div class="message-timestamp">
                        {{ msg.timestamp.strftime('%H:%M %p, %d %b') }}
                    </div>
                </div>
                {% else %}
                <div class="message bot-message">
                    <div class="bot-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div>
                        <div class="message-content">
                            <div class="message-text">{{ msg.message | safe }}</div>
                        </div>
                        <div class="message-timestamp">
                            {{ msg.timestamp.strftime('%H:%M %p, %d %b') }}
                        </div>
                    </div>
                </div>
                {% endif %}
                {% endfor %}
            </div>

            <div id="suggestions-mobile" class="suggestions-mobile">
                <div class="suggestion-chip">What is the cost of screen replacement?</div>
                <div class="suggestion-chip">Want to compare two devices?</div>
                <div class="suggestion-chip">Do you repair water-damaged phones?</div>
                <div class="suggestion-chip">Do you offer screen replacement services?</div>
            </div>

            <div class="input-container">
                <form id="chat-form" class="input-group" enctype="multipart/form-data">
                    <label for="image-upload" class="image-upload-label">
                        <i class="fas fa-image"></i>
                    </label>
                    <input type="file" id="image-upload" name="image" accept="image/*" style="display: none;">
                    <img id="image-preview" class="image-preview" alt="Image preview">
                    <input type="text" id="user_message" name="user_message" class="form-control"
                           placeholder="Ask a question" autocomplete="off">
                    <button type="submit" class="send-btn" id="send-btn">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        $(document).ready(function () {
            const $chatMessages = $('#chat-messages');
            const $form = $('#chat-form');
            const $input = $('#user_message');
            const $sendBtn = $('#send-btn');
            const $suggestionsDesktop = $('#suggestions-desktop');
            const $suggestionsMobile = $('#suggestions-mobile');
            const $imageInput = $('#image-upload');
            const $imagePreview = $('#image-preview');

            // Clear local storage on page load to ensure fresh session
            localStorage.removeItem('chatHistory');

            // Default suggestions from the design
            let suggestions = [
                "What is the cost of screen replacement?",
                "Want to compare two devices?",
                "Do you repair water-damaged phones?", 
                "What's your return policy?",
                "Do you offer screen replacement services?"
            ];

            // Handle suggestion click for both desktop and mobile
            $suggestionsDesktop.on('click', '.suggestion-chip', function () {
                const text = $(this).text();
                $input.val(text);
                $imageInput.val('');
                $imagePreview.hide();
                $form.submit();
            });

            $suggestionsMobile.on('click', '.suggestion-chip', function () {
                const text = $(this).text();
                $input.val(text);
                $imageInput.val('');
                $imagePreview.hide();
                $form.submit();
            });

            // Handle image upload preview
            $imageInput.on('change', function (e) {
                const file = e.target.files[0];
                if (file) {
                    if (!file.type.startsWith('image/')) {
                        alert('Please upload an image file.');
                        $imageInput.val('');
                        $imagePreview.hide();
                        return;
                    }
                    if (file.size > 5 * 1024 * 1024) {
                        alert('Image size must be less than 5MB.');
                        $imageInput.val('');
                        $imagePreview.hide();
                        return;
                    }
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        $imagePreview.attr('src', e.target.result).show();
                    };
                    reader.readAsDataURL(file);
                } else {
                    $imagePreview.hide();
                }
            });

            $form.submit(function (e) {
                e.preventDefault();
                const userMessage = $input.val().trim();
                const imageFile = $imageInput[0].files[0];
                if (!userMessage && !imageFile) return;

                const formData = new FormData();
                formData.append('user_message', userMessage);
                if (imageFile) {
                    formData.append('image', imageFile);
                }

                const now = new Date();
                const formattedTime = formatTime(now);
                
                let messageHtml = `
                    <div class="message user-message">
                `;
                if (imageFile) {
                    messageHtml += `<img src="${$imagePreview.attr('src')}" alt="User uploaded image" class="message-image">`;
                }
                if (userMessage) {
                    messageHtml += `
                        <div class="message-content">
                            <div class="message-text">${userMessage}</div>
                        </div>
                    `;
                }
                messageHtml += `
                        <div class="message-timestamp">${formattedTime}</div>
                    </div>
                `;
                $chatMessages.append(messageHtml);

                $chatMessages.append(`
                    <div class="message bot-message" id="loading">
                        <div class="bot-icon">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="loading-dots"><span></span><span></span><span></span></div>
                    </div>
                `);
                $chatMessages.scrollTop($chatMessages[0].scrollHeight);
                $sendBtn.prop('disabled', true);

                $.ajax({
                    type: "POST",
                    url: "{{ url_for('chatbot') }}",
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function (response) {
                        $('#loading').remove();
                        const markdownResponse = marked.parse(response.bot_response);
                        const sanitizedResponse = DOMPurify.sanitize(markdownResponse);
                        $chatMessages.append(`
                            <div class="message bot-message">
                                <div class="bot-icon">
                                    <i class="fas fa-robot"></i>
                                </div>
                                <div>
                                    <div class="message-content">
                                        <div class="message-text">${sanitizedResponse}</div>
                                    </div>
                                    <div class="message-timestamp">${formatTime(new Date(response.timestamp))}</div>
                                </div>
                            </div>
                        `);
                        $chatMessages.scrollTop($chatMessages[0].scrollHeight);
                        saveChatHistory();
                        
                        // Keep using the existing suggestion functionality
                        if (typeof updateSuggestions === 'function') {
                            updateSuggestions(userMessage);
                        }
                    },
                    error: function () {
                        $('#loading').remove();
                        $chatMessages.append(`
                            <div class="message bot-message">
                                <div class="bot-icon">
                                    <i class="fas fa-robot"></i>
                                </div>
                                <div>
                                    <div class="message-content">
                                        <div class="message-text">Sorry, something went wrong. Please try again.</div>
                                    </div>
                                    <div class="message-timestamp">${formatTime(new Date())}</div>
                                </div>
                            </div>
                        `);
                        $chatMessages.scrollTop($chatMessages[0].scrollHeight);
                        saveChatHistory();
                    },
                    complete: function () {
                        $sendBtn.prop('disabled', false);
                        $input.val('');
                        $imageInput.val('');
                        $imagePreview.hide();
                    }
                });
            });

            function saveChatHistory() {
                localStorage.setItem('chatHistory', $chatMessages.html());
            }

            function formatTime(date) {
                return date.toLocaleString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: true,
                    day: '2-digit',
                    month: 'short'
                });
            }

            // Fade out flash messages
            setTimeout(() => {
                $('.alert').each(function () {
                    $(this).css('transition', 'opacity 0.5s ease').css('opacity', '0');
                    setTimeout(() => $(this).remove(), 500);
                });
            }, 5000);
            
            // Keep the existing dynamic suggestion functionality
            if (typeof updateSuggestions === 'function') {
                // Initial suggestions
                updateSuggestions("");
            }
        });
    </script>
</body>
</html>