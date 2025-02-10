# Beer Rating Prediction - Machine Learning Model

## Project Overview
This project builds a **Machine Learning model** to predict the **overall rating of a beer** based on various features such as beer style, ABV, and user reviews. The dataset includes beer-related attributes and user reviews, and we process textual data using **TF-IDF vectorization**. Two models are implemented:
1. **Random Forest Regressor**
2. **Linear Regression**

## Dataset Details
The dataset used is `train.csv`, which contains the following key fields:
- `beer/ABV`: Alcohol by Volume of the beer.
- `beer/name`: Name of the beer.
- `beer/style`: The style/type of beer.
- `review/appearance`: Rating of beer's appearance (1.0 - 5.0).
- `review/aroma`: Rating of beer's aroma (1.0 - 5.0).
- `review/overall`: **Target variable** (Beer’s overall rating from 1.0 to 5.0).
- `review/text`: The text review given by a user.

## Workflow
### **1️ Data Preprocessing**
- **Dropped unnecessary columns** such as user profile, timestamps, and brewer ID.
- **Encoded categorical features** (`beer/name`, `beer/style`) using Label Encoding.
- **Handled missing values** in `review/text` by replacing them with an empty string.

### **2️ Feature Engineering**
- Applied **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform the `review/text` column into a numerical format.

### **3️ Model Training**
- **Splitting Data**: Used an **80-20 split** for training and testing respectively.
- **Implemented Two Models**:
  - **Random Forest Regressor**
  - **Linear Regression**

### **4️ Model Evaluation**
- Evaluated models using:
  - **RMSE (Root Mean Squared Error)**
  - **R² Score (Coefficient of Determination)**
  - **Accuracy Estimate**

##  How to Run the Project
### **1️ Install Dependencies**
Make sure you have the required Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **2️ Run the Python Script**
Execute the main script (`random_forest_model.py/linaer_reg_model`) in Google Colab or any Python environment:
```bash
python random_forest_model.py
```

### **3️ Model Output**
After running the script, you will see:
- **RMSE, R² Score, and Accuracy**

## Model Accuracy

| **Random Forest --- 95.9%** 
| **Linear Regression --- 96%** 


## License
This project is open-source.
---


