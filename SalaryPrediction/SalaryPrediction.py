import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('Salary_Data.csv')

# Display the original DataFrame
print("Original Table:")
print(df)

# Drop duplicates from the DataFrame
df.drop_duplicates(inplace=True)
# Update table
print("\nUpdated DataFrame (duplicates removed):")
print(df)

# Check for NaN values in columns
print("\nNaN counts per column:")
print(df.isna().sum())

# Drop rows with any NaN values
df.dropna(inplace=True)
# Print updated DataFrame
print("\nUpdated DataFrame (NaN values removed):")
print(df)

# Check data types of columns
print("\nData types of columns:")
print(df.dtypes)

# Encode 'Gender' into 0 for Male and 1 for Female
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Display the DataFrame after encoding
print("\nDataFrame after encoding Gender:")
print(df)

# Encode 'Education Level' into numeric values using Label Encoding
label_encoder = LabelEncoder()
df['Education Level'] = label_encoder.fit_transform(df['Education Level'])

# Create a dictionary to store the mapping between original labels and encoded labels
education_level_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Display the DataFrame after encoding Education Level
print("\nDataFrame after encoding Education Level:")
print(df)

# Identify outliers using the InterQuartile Range (IQR) method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print outliers based on Salary
outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("\nOutliers based on Salary:")
print(outliers[['Salary']])

# Remove rows with outliers
df_cleaned = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]
# Filter the DataFrame to include only the relevant job titles
relevant_jobs = ['Manager', 'Marketing', 'Sales', 'Procurement']
df_filtered = df_cleaned[df_cleaned['Job Title'].isin(relevant_jobs)]

# Create box plots for each job title
plt.figure(figsize=(12, 8))
df_filtered.boxplot(column='Salary', by='Job Title')
plt.title('Salary Distribution by Job Title')
plt.suptitle('')  # This removes the automatic suptitle
plt.ylabel('Salary')
plt.xlabel('Job Title')
plt.show()

# Split the dataset into training and testing sets
X = df_cleaned[['Age', 'Gender', 'Education Level', 'Years of Experience']]
y = df_cleaned['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
regr = LinearRegression()
regr.fit(X_train, y_train)

# Predict for a new data point
new_education_level = education_level_mapping["Master's"]

predicted_salary = regr.predict([[29, 1, new_education_level, 2]])

# Round the predicted salary to the nearest thousand
predicted_salary_rounded = round(predicted_salary[0] / 1000) * 1000

print("\nPredicted Salary for a 29-year-old Female with Master's and 2 years of experience (rounded to the nearest thousand):")
print(predicted_salary_rounded)
