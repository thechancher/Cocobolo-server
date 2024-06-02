# used for manipulate files
import os
# used to calculate the execution time
import time
# used for configurate the server
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
# used for decode the image
import base64
# use to instance the Cocobolo class
from cocobolo import Cocobolo

# instance the server and config the entrance point
app = Flask(__name__, template_folder="template")
CORS(app)
# set as debug execution
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")
# instance the Cocobolo class
cocobolo = Cocobolo()

# GET: main entrance point to the web app
@app.route("/")
def main():
    # present the web app
    return render_template("index.html")

# GET: url to get the classes name that the model can predict
@app.route("/class_names")
def names():
    # response in JSON format
    return jsonify(cocobolo.getClassNames())

# POST: url to get the prediction based in the image given
@app.route("/ViT", methods=["POST"])
def receiveImage():
    # start the time
    start_time = time.time()
    # receive the data from POST
    data = request.get_json()
    # receive just the Base64 value
    image = data["image"]
    # response as JSON
    response = {}
    
    try:
        # save the image received
        with open("/home/thechancher/ViT/images/image.jpg", "wb") as fh:
            fh.write(base64.urlsafe_b64decode(image))
        # predict the class based in the image given
        probabilities = cocobolo.predict()
        # pack the probabilities in the response
        response["probs"] =  probabilities 
    except BaseException as e:
        # pack the error in the response
        print("Error:", e)
        response["error"] = e
    
    # calculate the execution time
    elapsed_time = time.time() - start_time
    # pack the execution time in the response
    response["time"] = elapsed_time
    # response in JSON format
    return jsonify(response)

if __name__ == "__main__":
    # used for use only HTTP
    # app.run(host="0.0.0.0", port=80, debug=True)
    
    # port: 443 = used for HTTPS
    # ssl_context = used for present the SSL certificates
    app.run(host="0.0.0.0", debug=True, port=443, ssl_context=("/home/thechancher/ViT/cert.pem", "/home/thechancher/ViT/key.pem"))