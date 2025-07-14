import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load cleaned, filled dataset
df = pd.read_csv('data/Typhoid_Fever_data_filled_manual.csv')

# Drop rows with missing target
df = df.dropna(subset=['Symptoms Severity'])

# Features and target
features = [
    'Age', 'Gender', 'Hemoglobin (g/dL)', 'Platelet Count',
    'Calcium (mg/dL)', 'Potassium (mmol/L)',
    'Blood Culture Bacteria', 'Urine Culture Bacteria'
]
X = df[features]
y = df['Symptoms Severity']

# Encode categorical variables
categorical_cols = ['Gender', 'Blood Culture Bacteria', 'Urine Culture Bacteria']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Impute missing values (if any)
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode target labels
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Evaluation
print("✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model and encoders for future use
joblib.dump(knn, "models/knn_typhoid_model.pkl")
joblib.dump(scaler, "models/scaler_knn.pkl")
joblib.dump(y_encoder, "models/label_encoder_knn.pkl")
for col, le in le_dict.items():
    joblib.dump(le, f"models/label_encoder_{col}_knn.pkl")

print("\n✅ KNN model and encoders saved successfully.")
