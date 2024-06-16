import os
import time
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.llms import Replicate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

#assistant_prompt = "{query_str}"

#query_wrapper_prompt = SimpleInputPrompt(assistant_prompt)

#model_config = {'protected_namespaces': ()}

#try:
    # Setup CTransformers LLM
#    llm = CTransformers(
#       model="a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea#"),
#      tokenizer = AutoTokenizer.from_pretrained,
#      auth_token=HF_TOKEN,
#       model_type="llama",
 #       config={'max_new_tokens': 256,
 #               'temperature': 0.8, 
 #               'top_k': 50},
 #       force_download=True,
 #       **model_config# Adjusted for performance
 #   )
#except EnvironmentError as e:
#  print(f"Error loading model: {e}")
#    model = None

embeddings = download_hugging_face_embeddings()
if embeddings is None:
    raise ValueError("The embeddings is None. Please check the download_hugging_face_embeddings function.")
print(f"Embeddings: {embeddings}")

os.environ["REPLICATE_API_TOKEN"] = "r8_XiRFc4A1H54ShmMiGgT7Kx5BrCLHFwx3ozU2b"

# Initialize the Replicate model
llm = Replicate(
    model="a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
    config={
        'max_new_tokens': 100,  # Maximum number of tokens to generate in response
        'temperature': 0.7,     # Optimal temperature for balanced randomness and coherence
        'top_k': 50             # Optimal top-k value for considering the top 50 predictions
    }
)
# Flask routes
@app.route("/")
def index():
    return render_template('bot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")

        # Display spinner
        result = {"generated_text": "Thinking..."}

        # Simulate processing delay
        time.sleep(1)

        # Retrieve response from the model
        result = llm.generate([input_text])
        print(f"LLMResult: {result}")

        # Access the generated text from the result object
        if result.generations and result.generations[0]:
            generated_text = result.generations[0][0].text
        else:
            generated_text = "No response generated."

        print(f"Response: {generated_text}")

        return str(generated_text)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)