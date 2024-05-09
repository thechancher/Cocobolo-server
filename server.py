import os

from dotenv import load_dotenv

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

import base64

from cocobolo import Cocobolo

load_dotenv()

app = Flask(__name__, template_folder="template")
CORS(app)

app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")

cocobolo = Cocobolo()

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/names")
def names():
    names = {
        "index": [],
        "names": cocobolo.getNames()
    }
    return jsonify(names)

@app.route("/CNN", methods=["POST"])
def receiveImage():
    data = request.get_json()
    # receive just the Base64 value
    image = data["image"]
    # response a JSON
    response = {}
    
    try:
        with open("/home/thechancher/ViT/images/image.jpg", "wb") as fh:
            fh.write(base64.urlsafe_b64decode(image))
        probabilities = cocobolo.predict()
        
        response["probs"] =  probabilities 
    except BaseException as e:
        print("Error:", e)

    return jsonify(response)

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=80, debug=True)
    app.run(host="0.0.0.0", debug=True, port=443, ssl_context=("/home/thechancher/ViT/cert.pem", "/home/thechancher/ViT/key.pem"))