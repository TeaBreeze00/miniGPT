# miniGPT Web Interface

This project provides a web interface for text generation inspired by the miniGPT character-level RNN model. It allows users to enter a prompt and receive generated text continuations.

## Important Note

To run this please do the following:

1. Set up a separate Python API service (using Flask, FastAPI, etc.) to handle model inference
2. Use ONNX.js or TensorFlow.js to run the model directly in JavaScript
3. Use a serverless function specifically designed for ML inference (like AWS Lambda with custom runtime)

## Setup Instructions

1. **Install Dependencies**
   \`\`\`
   npm install
   # or
   yarn install
   \`\`\`

2. **Run the Development Server**
   \`\`\`
   npm run dev
   # or
   yarn dev
   \`\`\`

3. **Access the Application**
   Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

- `app/page.tsx`: Main UI component with the prompt input form
- `app/api/generate/route.ts`: API endpoint that handles text generation requests (mock implementation)

## Usage

1. Enter a prompt in the text field (e.g., "The weather today is")
2. Click "Generate Text" to get a continuation
3. The generated text will appear below the form

## Integrating with Your Actual PyTorch Model

To use your actual trained model, you would need to:

1. Create a separate Python API service using Flask or FastAPI
2. Expose an endpoint that loads your model and generates text
3. Update the frontend to call this external API instead of the mock implementation

Example Python API service (not included):
\`\`\`python
from flask import Flask, request, jsonify
import torch
# Import your model definition and utilities

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    length = data.get('length', 200)
    
    # Load your model and generate text
    # ...
    
    return jsonify({'generatedText': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
\`\`\`

