from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('temperature_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
data= pd.read_csv('C:/Users/User/OneDrive/Desktop/Weather_forest_analysis/Weather_Forest_Dataset.csv')

# Calculate average values for rain and forest_area
average_rain = data['rain'].mean()
average_forest_area = data['forest_area'].mean()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    year = int(form_data['year'])
    month = int(form_data['month'])
    
    # Check if the year is in the future
    if year > 2023:
        # Use average values for rain and forest_area for future dates
        rainfall = average_rain
        forest_area = average_forest_area
    else:
        # Get additional features from historical data
        filtered_data = data[(data['Year'] == year) & (data['Month'] == month)]
        if not filtered_data.empty:
            rainfall = filtered_data['rain'].values[0]
            forest_area = filtered_data['forest_area'].values[0]
        else:
            return jsonify({'error': 'No data found for the given year and month'}), 400

    # Prepare the input array
    input_features = np.array([[year, month, rainfall, forest_area]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    return jsonify({'temperature': prediction})

if __name__ == '__main__':
    app.run(debug=True)
