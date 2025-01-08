import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r'C:\Users\Harshlaugale\Downloads\Heart_Disease_Prediction_ML\Heart_Disease_Prediction.csv')

# Convert categorical target variable to numeric
label_encoder = LabelEncoder()
df['Heart Disease'] = label_encoder.fit_transform(df['Heart Disease'])  # Convert 'Presence'/'Absence' to 1/0

# Preprocessing the data
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# Split the dataset into features (X) and target (y)
X = df[['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'Max HR', 'Exercise angina']]  # Reduced features
y = df['Heart Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# Evaluate the models
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_dec_tree = dec_tree.predict(X_test)

log_reg_acc = accuracy_score(y_test, y_pred_log_reg) * 100  # Convert to percentage
dec_tree_acc = accuracy_score(y_test, y_pred_dec_tree) * 100  # Convert to percentage

# Choose the model with higher accuracy
if log_reg_acc > dec_tree_acc:
    chosen_model = "Logistic Regression"
    chosen_model_pred = log_reg.predict
    chosen_model_acc = log_reg_acc
else:
    chosen_model = "Decision Tree"
    chosen_model_pred = dec_tree.predict
    chosen_model_acc = dec_tree_acc

# Define Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting data from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    chest_pain = int(request.form['chest_pain'])
    cholesterol = int(request.form['cholesterol'])
    max_hr = int(request.form['max_hr'])
    exercise_angina = int(request.form['exercise_angina'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'Chest pain type': [chest_pain],
        'Cholesterol': [cholesterol], 'Max HR': [max_hr],
        'Exercise angina': [exercise_angina]
    })

    # Standardize the input data for Logistic Regression
    input_data_scaled = scaler.transform(input_data)

    # Prediction from the chosen model
    prediction = chosen_model_pred(input_data_scaled)[0]
    
    # Determine if heart disease is present
    result = "Yes" if prediction == 1 else "No"

    # Render the results page
    return render_template('result.html',
                           result=result,
                           chosen_model=chosen_model,
                           chosen_model_acc=f"{chosen_model_acc:.2f}%",  # Format accuracy as percentage
                           log_reg_acc=f"{log_reg_acc:.2f}%",  # Format as percentage
                           dec_tree_acc=f"{dec_tree_acc:.2f}%")  # Format as percentage

if __name__ == '__main__':
    app.run(debug=True)
