# app.py

import os
import logging
from flask import Flask, request, jsonify
from roboflow import Roboflow
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv() 

# Set up basic logging to see informational messages and errors in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask web application
app = Flask(__name__)
# Create a folder to temporarily store uploaded images
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- Roboflow Initialization ---
model = None
try:
    # Get credentials from environment variables
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_WORKSPACE_ID = os.getenv("ROBOFLOW_WORKSPACE_ID")
    ROBOFLOW_MODEL_ID = "cow-and-buffalo-ikxjv"
    ROBOFLOW_MODEL_VERSION = 2

    # Check if the environment variables are loaded
    if not all([ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE_ID]):
        raise ValueError("Roboflow API Key or Workspace ID is missing in the .env file.")

    # Initialize the Roboflow client and load the model
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE_ID).project(ROBOFLOW_MODEL_ID)
    model = project.version(ROBOFLOW_MODEL_VERSION).model
    logging.info("✅ Successfully connected to Roboflow model.")

except Exception as e:
    # Log any errors that occur during initialization
    logging.error(f"❌ Error initializing Roboflow: {e}")


# --- API Route for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    # If the model failed to load, return an error
    if model is None:
        return jsonify({"error": "Model is not initialized. Check server logs for details."}), 500

    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        # Sanitize the filename and create a path to save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the file to the server
            file.save(filepath)
            logging.info(f"Image received and saved to {filepath}")

            # Perform prediction on the saved image
            logging.info("🧠 Sending image to Roboflow for prediction...")
            prediction_result = model.predict(filepath, confidence=40, overlap=30).json()
            logging.info(f"✅ Prediction received: {prediction_result}")

            predictions = prediction_result.get('predictions', [])
            
            if not predictions:
                # If the model did not detect any objects
                response_data = {
                    "prediction": "Nothing detected",
                    "confidence": 0,
                    "tags": ["no-detection"]
                }
            else:
                # Find the detected object with the highest confidence score
                best_prediction = max(predictions, key=lambda p: p['confidence'])
                
                # Format the response to match what the frontend expects
                response_data = {
                    "prediction": best_prediction['class'].title(), # Capitalize class name
                    "confidence": best_prediction['confidence'],
                    "tags": list(set([p['class'] for p in predictions])) # List unique detected classes
                }
            
            return jsonify(response_data)

        except Exception as e:
            logging.error(f"❌ An error occurred during prediction: {e}")
            return jsonify({"error": "Failed to process the image.", "details": str(e)}), 500
        
        finally:
            # IMPORTANT: Clean up and delete the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Cleaned up file: {filepath}")
            
    return jsonify({"error": "An unknown error occurred."}), 500

# --- Start the Server ---
if __name__ == '__main__':
    # Run the Flask app on port 5000, accessible from other devices on the network
    app.run(host='0.0.0.0', port=5000)
