import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned, filled dataset
df = pd.read_csv('data/Typhoid_Fever_data_filled_manual.csv')

# Remove rows with missing severity for clean visuals
df = df.dropna(subset=['Symptoms Severity'])

# Set consistent style
sns.set(style="whitegrid")

# 1️⃣ Count of Severity Cases 
plt.figure(figsize=(7, 5))
sns.countplot(x='Symptoms Severity', data=df, palette='coolwarm')
plt.title('Count of Cases by Symptoms Severity')
plt.xlabel('Symptoms Severity')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.show()

# 2️⃣ Gender-wise Severity Distribution 
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Symptoms Severity', data=df, palette='viridis')
plt.title('Gender-wise Symptoms Severity Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Patients')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()

# 3️⃣ Blood Culture Result vs Symptoms Severity 
plt.figure(figsize=(10, 6))
sns.countplot(y='Blood Culture Bacteria', hue='Symptoms Severity', data=df, palette='Set2')
plt.title('Blood Culture Result vs Symptoms Severity')
plt.xlabel('Number of Patients')
plt.ylabel('Blood Culture Bacteria')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()
