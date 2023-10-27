from flask import Flask, request, jsonify
import torch as torch
import numpy as np
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


app = Flask(__name__)

tokenizer1 = RobertaTokenizer.from_pretrained("roberta-base")

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)



@app.route("/")
def home():
    return "<h2>Comment sentiment analysis based on RoBERTa</h2>"


@app.route("/analyze_sent", methods=["POST"])
def analyze_sent():
     
    print(request.json['name'])
    
    encoded_text = tokenizer(request.json['name'], return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    scores_dict = {
        "positive": scores[2],
        "neutral": scores[1],
        "negative": scores[0]
    }
    scores_dict = {k: float(v) for k, v in scores_dict.items()}
    scores_dict["good"] = bool(np.argmax(scores))

    print(scores_dict)

    return jsonify(scores_dict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8354, debug=True)