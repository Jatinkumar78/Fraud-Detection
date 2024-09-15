
# Fraud Detection Assignment

## Folder Structure:
```
fraud_detection/
│
├── dataset/
│   └── credit_card_transactions.csv   # Place the dataset here
│
├── src/
│   ├── eda.py                         # Code for EDA
│   ├── preprocessing.py                # Code for data preprocessing
│   ├── model.py                        # Code for model building and evaluation
│   ├── app.py                          # Flask API (optional)
│   └── requirements.txt                # Dependencies (Flask, scikit-learn, etc.)
│
├── outputs/
│   └── confusion_matrix.png            # Output of the confusion matrix
│   └── feature_importance.png          # Feature importance graph
│
└── README.md                           # Instructions for your examiner
```

### Instructions:
1. **Dataset**: Place your `credit_card_transactions.csv` in the `dataset/` folder.
2. **Run EDA**: Run `src/eda.py` to perform exploratory data analysis.
3. **Preprocess Data**: Run `src/preprocessing.py` to clean and prepare the data.
4. **Model Building**: Execute `src/model.py` to train and evaluate the model.
5. **API Hosting (Optional)**: Use `src/app.py` to host the model using Flask.

Happy Coding!
