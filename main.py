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
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No data provided."}), 400
        
        user_message = data.get('user_message', '').strip()
        session_id = request.cookies.get('session_id')
        
        # Get chat history for context-aware suggestions
        messages = []
        if chat_history_collection is not None and session_id:
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