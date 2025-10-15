# app.py (Final Code for Render Deployment)

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- FIXED IMPORTS for LangChain Core ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
from werkzeug.middleware.proxy_fix import ProxyFix
# --- END FIXED IMPORTS ---

# Load environment variables (for local testing; Render loads securely)
load_dotenv() 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # This check ensures the app doesn't run without a key
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in Render.")

# --- Pydantic Schemas for Structured Output ---

class TaskAnalysis(BaseModel):
    """Schema for structured output of task details."""
    text: str = Field(description="The cleaned-up final task text.")
    time: str = Field(description="Extracted deadline or date. Must be YYYY-MM-DD or a time of day/unspecified.")
    category: str = Field(description="A category label (e.g., Work, Study, Health).")
    urgent: bool = Field(description="True if the task is marked urgent or uses words like ASAP.")
    note: str = Field(description="A helpful, concise note or warning for the user.")
    effort_score: str = Field(description="Assigned score: Low, Medium, High, or Critical.", 
                              enum=["Low", "Medium", "High", "Critical"])

class SuggestionList(BaseModel):
    """Schema for structured output of suggested tasks."""
    suggestions: List[str] = Field(description="A list of 5 complete task suggestions.")


# --- INITIALIZE FLASK AND GEMINI ---
app = Flask(__name__)

# VITAL FIX: Apply ProxyFix to resolve Render 404 routing issues
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1) 

CORS(app) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=GEMINI_API_KEY
)

def get_today_string():
    return datetime.now().strftime("%Y-%m-%d")


# --- ROUTE 1: Serves the Frontend HTML (Necessary for a combined deployment) ---
# NOTE: This assumes you have moved your index.html into a folder named /static
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    return send_from_directory('static', path)


# --- ROUTE 2: /api/analyze (Task Analysis) ---
@app.route("/api/analyze", methods=["POST"])
def analyze_handler():
    try:
        data = request.get_json()
        task_text = data.get("task_text")
        current_date_string = get_today_string()

        if not task_text:
            return jsonify({"error": "Missing task_text"}), 400

        # LangChain Setup for Analysis Chain
        parser = JsonOutputParser(pydantic_object=TaskAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             f"You are a professional task analysis engine. The current date is {current_date_string}. "
             "Your sole purpose is to return ONLY a valid JSON object. "
             "Strictly format any date found as YYYY-MM-DD. \n"
             "{format_instructions}" # This is the placeholder for the parser's instructions
            ),
            ("user", "{user_input}"),
        ])
        
        # FIX: Inject the formatting instructions into the prompt template using .partial()
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | llm | parser
        
        # Invoke the LLM, supplying the variable the prompt expects
        result = chain.invoke({"user_input": task_text})

        return jsonify(result)

    except ValidationError as e:
        print(f"Validation Error: {e}")
        return jsonify({"error": "AI returned invalid JSON structure.", "details": str(e)}), 500
    except Exception as e:
        print(f"ERROR in analyze.py: {e}")
        return jsonify({"error": "Internal Server Error during AI analysis."}), 500


# --- ROUTE 3: /api/suggest (Dynamic Suggestions) ---
@app.route("/api/suggest", methods=["POST"])
def suggest_handler():
    try:
        data = request.get_json()
        partial_task = data.get("partial_task")

        if not partial_task:
            return jsonify({"error": "Missing partial_task"}), 400

        # LangChain Setup for Suggestions Chain
        parser = JsonOutputParser(pydantic_object=SuggestionList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful AI completer. Generate 5 unique suggestions to complete the user's partial text. "
             f"The response format must be:\n"
             "{format_instructions}"
            ),
            ("user", "{user_input}"),
        ])
        
        # FIX: Inject the formatting instructions into the prompt template using .partial()
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        
        result = chain.invoke({"user_input": partial_task})

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in suggest.py: {e}")
        return jsonify({"error": "Internal Server Error during suggestion generation."}), 500


# --- STARTUP COMMAND FOR RENDER (Gunicorn) ---
if __name__ == '__main__':
    # This block is used for local development only.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
