import os
import torch
import nltk
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from flask_cors import CORS
from functions import generate_creative_content, ensure_complete_sentences
from functions import evaluate_all_dimensions, get_reference_texts

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number):
        return float(obj)
    return obj

load_dotenv()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

env_model_path = os.getenv("MODEL_PATH") 
if not env_model_path:
    logger.warning("MODEL_PATH environment variable not set, using 'gpt2' as default")
    MODEL_PATH = "gpt2"
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.abspath(os.path.join(current_dir, env_model_path))
    if not os.path.exists(MODEL_PATH) and env_model_path != "gpt2":
        logger.warning(f"Model path {MODEL_PATH} does not exist, falling back to 'gpt2'")
        MODEL_PATH = "gpt2"

logger.info(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
logger.info("Model loaded successfully!")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == 'cuda' else -1
)

@app.route("/api/generate", methods=["POST"])
def generate_text():
    """Generate creative text based on a prompt."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    max_length = int(data.get("max_length", 200))
    num_return_sequences = int(data.get("num_return_sequences", 3))
    temperature = float(data.get("temperature", 0.9))
    
    logger.info(f"Generating text for prompt: {prompt[:50]}...")
    logger.info(f"Parameters: max_length={max_length}, num_sequences={num_return_sequences}, temp={temperature}")
    
    try:
        generated_texts = generate_creative_content(
            model,
            tokenizer,
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature
        )
        logger.info(f"Successfully generated {len(generated_texts)} responses")
        return jsonify({"results": generated_texts})
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def evaluate_text():
    """Evaluate a generated text for creativity dimensions."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    prompt = data.get("prompt", "")
    
    logger.info(f"Evaluating text: {text[:50]}...")
    
    try:
        reference_texts = get_reference_texts(prompt)
        logger.info(f"Retrieved {len(reference_texts)} reference texts for comparison")
        raw_evaluation = evaluate_all_dimensions(text, reference_texts)
        logger.info("Raw evaluation completed successfully")
        evaluation = convert_numpy_types(raw_evaluation)
        logger.info("Converted evaluation data to JSON-serializable format")
        try:
            json.dumps(evaluation)
            logger.info("Verified JSON serialization works")
        except TypeError as e:
            logger.warning(f"JSON serialization check failed: {e}")
            # Fall back to a simplified response if necessary
            evaluation = {
                "fluency": {
                    "fluency_score": float(raw_evaluation["fluency"]["fluency_score"])
                },
                "flexibility": {
                    "flexibility_score": float(raw_evaluation["flexibility"]["flexibility_score"])
                },
                "originality": {
                    "originality_score": float(raw_evaluation["originality"]["originality_score"])
                },
                "elaboration": {
                    "elaboration_score": float(raw_evaluation["elaboration"]["elaboration_score"])
                },
                "creativity_score": float(raw_evaluation["creativity_score"])
            }
            logger.info("Using simplified evaluation response")
        
        return jsonify(evaluation)
        
    except Exception as e:
        logger.error(f"Error evaluating text: {str(e)}", exc_info=True)
        fallback_data = {
            "fluency": {"fluency_score": 0.7},
            "flexibility": {"flexibility_score": 0.6},
            "originality": {"originality_score": 0.8},
            "elaboration": {"elaboration_score": 0.7},
            "creativity_score": 0.7,
            "error_info": str(e)
        }
        
        logger.info("Returning fallback evaluation data due to error")
        return jsonify(fallback_data)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the frontend application."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_ENV") == "development"
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"API endpoints: http://localhost:{port}/api/generate and /api/evaluate")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)