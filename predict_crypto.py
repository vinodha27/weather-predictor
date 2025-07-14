

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.read_csv('weather.csv')

# Step 2: Define input (features) and output (target)
X = data[['humidity', 'wind_speed']]  # Features
y = data['temperature']  # Target

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Tamil Nadu districts weather data (sample)
places_data = {
    "ariyalur": {"humidity": 65, "wind_speed": 5},
    "chengalpattu": {"humidity": 70, "wind_speed": 6},
    "chennai": {"humidity": 70, "wind_speed": 5},
    "coimbatore": {"humidity": 65, "wind_speed": 6},
    "cuddalore": {"humidity": 68, "wind_speed": 5},
    "dharmapuri": {"humidity": 60, "wind_speed": 6},
    "dindigul": {"humidity": 60, "wind_speed": 5},
    "erode": {"humidity": 57, "wind_speed": 6},
    "kallakurichi": {"humidity": 63, "wind_speed": 5},
    "kanchipuram": {"humidity": 68, "wind_speed": 5},
    "karur": {"humidity": 62, "wind_speed": 6},
    "krishnagiri": {"humidity": 58, "wind_speed": 6},
    "madurai": {"humidity": 60, "wind_speed": 5},
    "nagapattinam": {"humidity": 75, "wind_speed": 5},
    "namakkal": {"humidity": 60, "wind_speed": 6},
    "perambalur": {"humidity": 62, "wind_speed": 5},
    "pudukkottai": {"humidity": 61, "wind_speed": 5},
    "ramanathapuram": {"humidity": 70, "wind_speed": 4},
    "ranipet": {"humidity": 65, "wind_speed": 6},
    "salem": {"humidity": 58, "wind_speed": 6},
    "sivaganga": {"humidity": 60, "wind_speed": 5},
    "thanjavur": {"humidity": 63, "wind_speed": 4},
    "theni": {"humidity": 60, "wind_speed": 5},
    "thoothukudi": {"humidity": 68, "wind_speed": 5},
    "tiruchirappalli": {"humidity": 62, "wind_speed": 4},
    "tirunelveli": {"humidity": 65, "wind_speed": 5},
    "tirupattur": {"humidity": 58, "wind_speed": 6},
    "tiruppur": {"humidity": 60, "wind_speed": 5},
    "tiruvallur": {"humidity": 70, "wind_speed": 6},
    "tiruvannamalai": {"humidity": 60, "wind_speed": 5},
    "tiruvarur": {"humidity": 65, "wind_speed": 5},
    "vellore": {"humidity": 55, "wind_speed": 7},
    "viluppuram": {"humidity": 65, "wind_speed": 5},
    "virudhunagar": {"humidity": 62, "wind_speed": 5}
}

# Step 5: Predict on test set and show results
predictions = model.predict(X_test)
print("Actual vs Predicted temperatures on test data:")
for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual}°C, Predicted: {predicted:.2f}°C")

# Step 6: Predict based on user input district
district = input("\nEnter a Tamil Nadu district to predict temperature: ").lower()

if district in places_data:
    hum = places_data[district]['humidity']
    wind = places_data[district]['wind_speed']
    custom_input = pd.DataFrame([[hum, wind]], columns=['humidity', 'wind_speed'])
    predicted_temp = model.predict(custom_input)
    print(f"Predicted Temperature in {district.title()}: {predicted_temp[0]:.2f}°C")
else:
    print("Sorry, data for this district is not available.")
