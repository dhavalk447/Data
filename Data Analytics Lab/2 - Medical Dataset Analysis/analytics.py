import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Generate a random medical dataset
np.random.seed(42)

# Simulating a dataset with 1000 entries
n_samples = 1000
ages = np.random.randint(18, 90, n_samples)  # Random ages between 18 and 90
genders = np.random.choice(['Male', 'Female'], size=n_samples)  # Random gender
blood_pressure = np.random.randint(90, 180, n_samples)  # Random blood pressure values
cholesterol = np.random.randint(150, 300, n_samples)  # Random cholesterol values
diabetes = np.random.choice([0, 1], size=n_samples)  # 0: No, 1: Yes (for diabetes)

# Create a DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Blood Pressure': blood_pressure,
    'Cholesterol': cholesterol,
    'Diabetes': diabetes
})

# Step 2: Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

print("\nSummary Statistics:")
print(data.describe())

# Count the number of diabetes cases
print("\nDiabetes case distribution:")
print(data['Diabetes'].value_counts())

# Step 3: Data Visualization
# Visualizing the relationship between age and blood pressure
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Blood Pressure', data=data, hue='Diabetes', palette='coolwarm')
plt.title('Age vs Blood Pressure (Diabetes Status)')
plt.show()

# Step 4: Data Preprocessing
# Convert categorical columns (Gender) to numerical values
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Standardize numerical features
scaler = StandardScaler()
data[['Age', 'Blood Pressure', 'Cholesterol']] = scaler.fit_transform(data[['Age', 'Blood Pressure', 'Cholesterol']])

# Step 5: Train a model to predict diabetes
X = data[['Age', 'Gender', 'Blood Pressure', 'Cholesterol']]  # Features
y = data['Diabetes']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.show()
