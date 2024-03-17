from flask import Flask, request, jsonify
from BrawlerClassifierModel import BrawlerClassifierModel

app = Flask(__name__)
model = BrawlerClassifierModel()

@app.route('/train')
def train():
    dataset = request.args.get('dataset')
    return model.train(dataset)

@app.route('/predict')
def predict():
    dataset = request.args.get('dataset')
    return model.predict(dataset)

if __name__ == "__main__":
    app.run(debug=True)