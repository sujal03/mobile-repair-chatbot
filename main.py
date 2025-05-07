from flask import Flask, render_template, request, jsonify, make_response
from chatbot import init_components, generate_llm_response, generate_suggestions
import os
import pymongo
from dotenv import load_dotenv
import uuid
from datetime import datetime
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# MongoDB setup
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["FAQ_chatbot"]
chat_history_collection = mongo_db["chat_history"]

# Initialize database with index
def init_database():
    try:
        chat_history_collection.create_index("session_id", unique=True)
        logger.info("Database index created successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

init_database()

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    # Always create a new session_id for GET requests (page reload)
    session_id = generate_session_id()
    logger.info(f"New session created: {session_id}")

    # Initialize components with session_id
    vectordb, _, llm = init_components(session_id)

    if request.method == 'POST':
        try:
            user_message = request.form.get('user_message', '').strip()
            image_file = request.files.get('image')
            if not user_message and not image_file:
                return jsonify({"error": "Please provide a message or an image."}), 400

            # Validate image file
            if image_file:
                if not image_file.content_type.startswith('image/'):
                    return jsonify({"error": "Invalid file type. Please upload an image."}), 400
                if image_file.content_length > 5 * 1024 * 1024:  # 5MB limit
                    return jsonify({"error": "Image size exceeds 5MB limit."}), 400

            # Fetch or initialize chat history
            history_doc = chat_history_collection.find_one({"session_id": session_id}) or {
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.utcnow()
            }
            messages = history_doc.get("messages", [])

            # Prepare user message entry
            user_message_entry = {
                "role": "user",
                "message": user_message if user_message else "Image uploaded for analysis",
                "timestamp": datetime.utcnow()
            }

            # Handle image storage (store base64 in MongoDB for simplicity)
            if image_file:
                image_data = BytesIO(image_file.read())
                import base64
                image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
                user_message_entry["image_base64"] = image_base64
                user_message_entry["image_url"] = f"data:image/jpeg;base64,{image_base64}"

            messages.append(user_message_entry)

            # Generate bot response
            bot_response = generate_llm_response(
                user_message,
                vectordb,
                llm,
                messages,
                image_file=BytesIO(image_data.getvalue()) if image_file else None
            )
            bot_message_entry = {
                "role": "bot",
                "message": bot_response,
                "timestamp": datetime.utcnow()
            }
            messages.append(bot_message_entry)

            # Update chat history in database
            chat_history_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "messages": messages,
                        "last_updated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"Message saved for session {session_id}")

            response = jsonify({
                "bot_response": bot_response,
                "user_message": user_message,
                "timestamp": bot_message_entry["timestamp"].isoformat()
            })

            # Set session_id cookie
            response.set_cookie('session_id', session_id, max_age=86400 * 30)  # 30 days

            return response

        except Exception as e:
            logger.error(f"Chatbot error: {str(e)}")
            return jsonify({"error": "An error occurred."}), 500

    # GET request: Render chat interface with empty history
    response = make_response(render_template('chatbot.html', chat_history=[]))
    response.set_cookie('session_id', session_id, max_age=86400 * 30)  # 30 days
    return response

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()
        user_message = data.get('user_message', '').strip()
        if not user_message:
            return jsonify({"error": "No user message provided."}), 400

        suggestions = generate_suggestions(user_message)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return jsonify({"error": "Failed to generate suggestions."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=7000, host='0.0.0.0')