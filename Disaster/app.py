from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # For loading the model

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define the label map
label_map = {0: 'None', 1: 'Storm', 2: 'Wildfire', 3: 'Earthquake', 4: 'Flood', 5: 'Tornado'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from input
        data = request.json

        # Create DataFrame with one row
        df = pd.DataFrame([data])

        # Predict using the loaded model
        prediction = model.predict(df)[0]

        # Map the prediction to the label
        result = label_map.get(prediction, 'Unknown')

        # Return the result
        return jsonify({'disaster_type': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
