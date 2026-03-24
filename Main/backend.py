from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
# Load the pre-trained model
model = tf.keras.models.load_model('Main/LSTM_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    
    # Convert the data to a numpy array
    input_data = data['input']
    predcition = model.predict(input_data)
    return jsonify({'prediction': predcition.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
# This code is a Flask application that serves a pre-trained LSTM model for legacy sea-level forecasting.