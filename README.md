# capstone_fraud_detection_pipeline

## Overview
This project demonstrates a scalable data engineering pipeline combined with machine learning to detect fraudulent financial transactions. The goal is to improve fraud detection accuracy while efficiently processing large volumes of transactional data.

## Project Highlights
- Built a scalable data pipeline for fraud detection
- Applied machine learning classification techniques
- Analyzed model performance with real-world constraints

## Business Problem
Financial institutions process millions of transactions daily, making fraud detection complex and time-sensitive. Traditional approaches struggle with scale and accuracy, leading to delayed detection and financial losses.

## Hypothesis
The use of a data engineering pipeline combined with a machine learning model will result in higher accuracy in fraud detection compared to using a machine learning model alone.

## Dataset
- Source: Kaggle Fraudulent Transactions Dataset
- Contains anonymized financial transaction data
- Features include:
  - Transaction type
  - Transaction amount
  - Account balances
  - Fraud labels

## Data Engineering Pipeline
The data pipeline includes:
- Data ingestion from CSV
- Data cleaning (removal of missing values and duplicates)
- Feature engineering (one-hot encoding of categorical variables)
- Data preparation for machine learning models

## Model
- Logistic Regression (primary model)
- Decision Tree (used for comparison)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- Confusion Matrix

## Results
- Accuracy: ~99.9%
- Precision: ~91%
- Recall: ~48%

## Key Findings
- The model performs well in identifying non-fraudulent transactions
- A significant number of fraudulent transactions are missed
- Recall is critical in fraud detection and should be prioritized

## Limitations
- Dataset is simulated and may not fully represent real-world fraud patterns
- Class imbalance impacts model performance
- Logistic regression may not capture complex fraud behaviors

## Recommendations
- Improve recall to reduce missed fraud
- Use more advanced models (e.g., Random Forest, Gradient Boosting)
- Implement real-time fraud detection systems

## Future Work
- Explore advanced machine learning algorithms
- Address class imbalance using resampling techniques
- Develop real-time data processing pipelines

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure
fraud-detection-pipeline/
├── src/
│ └── main.py
├── data/
│ └── Fraud.csv
├── README.md
## Author
Brandon Lehr  
MSDA Program, Western Governors University
