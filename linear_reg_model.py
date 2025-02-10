# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "/content/drive/MyDrive/train.csv"
df = pd.read_csv(file_path)

# Dropping unnecessary columns
df.drop(["index", "beer/beerId", "beer/brewerId", "review/timeStruct", "review/timeUnix",
         "user/birthdayRaw", "user/birthdayUnix", "user/profileName", "user/gender", "user/ageInSeconds"], 
        axis=1, inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
df["beer/name"] = label_encoder.fit_transform(df["beer/name"])
df["beer/style"] = label_encoder.fit_transform(df["beer/style"])

# Handling missing values in review text
df["review/text"] = df["review/text"].fillna("")

# Converting review text into numerical using TF-IDF
tfidf = TfidfVectorizer(max_features=200)  
tfidf_matrix = tfidf.fit_transform(df["review/text"]).toarray()

# Converting TF-IDF to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"word_{i}" for i in range(tfidf_matrix.shape[1])])

# removing original text column
df.drop("review/text", axis=1, inplace=True)

# merging TF-IDF features
df.reset_index(drop=True, inplace=True)
tfidf_df.reset_index(drop=True, inplace=True)

#Joining TF-IDF features with original dataframe
df = pd.concat([df, tfidf_df], axis=1)

# x- features and y-  traget variable
X = df.drop("review/overall", axis=1)
y = df["review/overall"]

# Splitting data into training and testing sets (80% for training 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Training Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Model Evaluation
rmse = mean_squared_error(y_test, y_pred)  # Use squared=False for RMSE
r2 = r2_score(y_test, y_pred)

# Performance Metrics
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Model Accuracy
accuracy = (1 - (rmse / y_test.mean())) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
