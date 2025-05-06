from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Blood group encoding
blood_group_mapping = {
    "A+": 0, "A-": 1, "B+": 2, "B-": 3,
    "O+": 4, "O-": 5, "AB+": 6, "AB-": 7
}

# Sample AI training data
symptoms = np.array([
    [120, blood_group_mapping["A+"], 1],
    [140, blood_group_mapping["B-"], 0],
    [110, blood_group_mapping["O+"], 1],
    [130, blood_group_mapping["AB-"], 0],
    [100, blood_group_mapping["A-"], 1]
])

conditions = ["Hypertension", "Normal", "Low BP", "Normal", "Low BP"]

# Train the model
model = DecisionTreeClassifier()
model.fit(symptoms, conditions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    blood_group = data.get('bloodGroup')
    bp = data.get('bp')

    if not blood_group or not bp:
        return jsonify({'error': 'Missing blood group or blood pressure'}), 400

    # Normalize blood group (add "+" if user just types "A" instead of "A+")
    if blood_group in ["A", "B", "O", "AB"]:
        blood_group += "+"
    
    if blood_group not in blood_group_mapping:
        return jsonify({'error': 'Invalid blood group. Please use A+, A-, B+, etc.'}), 400

    try:
        bp_value = int(bp)
    except ValueError:
        return jsonify({'error': 'Invalid blood pressure'}), 400

    user_input = np.array([[bp_value, blood_group_mapping[blood_group], 1]])
    prediction = model.predict(user_input)

    recommendations = {
        "Hypertension": ["Reduce salt intake", "Exercise regularly", "Monitor BP"],
        "Normal": ["Maintain a healthy diet", "Regular check-ups"],
        "Low BP": ["Stay hydrated", "Increase salt intake"]
    }

    result_condition = prediction[0]
    recs = recommendations.get(result_condition, ["Consult a doctor for more information"])

    return jsonify({'condition': result_condition, 'recommendations': recs})

if __name__ == '__main__':
    app.run(debug=True)
