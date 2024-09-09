import pandas as pd

# Create a DataFrame for future dates (e.g., 2024 to 2035)
future_dates = pd.DataFrame({
    'Year': [year for year in range(2024, 2036) for month in range(1, 13)],
    'Month': [month for year in range(2024, 2036) for month in range(1, 13)]
})

# Load historical data
data= pd.read_csv('C:/Users/User/OneDrive/Desktop/Weather_forest_analysis/Weather_Forest_Dataset.csv')

# Calculate average values for rain and forest_area
average_rain = data['rain'].mean()
average_forest_area = data['forest_area'].mean()

# Assign these average values to the future dates
future_dates['rain'] = average_rain
future_dates['forest_area'] = average_forest_area

print(future_dates)
