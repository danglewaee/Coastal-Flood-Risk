from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load mô hình LSTM
model = load_model('Main/lstm_model.h5')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, 240, 1)

        # Dự đoán với mô hình LSTM
        prediction = model.predict(input_data)

        return jsonify({'prediction': prediction[0][0]})  # Trả về dự đoán
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
