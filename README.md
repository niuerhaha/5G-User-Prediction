# 5G User Prediction

This is a simple machine learning project. We use Random Forest and LightGBM to predict if a user will use 5G or not.

## About this project

We are learning machine learning, so we write this code to practice. The code do some basic things:

1. Read data from train.csv file
2. Fill missing values (use mode for category columns, use median for number columns)
3. Change category columns to numbers with LabelEncoder
4. Split data to train set and test set (80% and 20%)
5. Train two models: Random Forest and LightGBM
6. Draw some pictures to compare which model is better

## Dataset

The data file is train.csv. It have these columns:

- id column: just a number for each row
- target column: 0 or 1, means this user use 5G or not
- cat_ columns: these are category type data
- num_ columns: these are number type data

We cannot upload the data file because it is too big (more than 200MB). If you want to run my code, you need to put your own train.csv file in this folder.

## How to use

First, install the things you need:

```
pip install -r requirements.txt
```

Then run the Python file:

```
python 5GUser-Prediction.py
```

After running, you will see some numbers printed in the terminal, and 4 pictures will be saved:

| Picture file | What inside |
|--------------|-------------|
| rf_feature_importance.png | Important features from Random Forest |
| lgb_feature_importance.png | Important features from LightGBM |
| roc_comparison.png | ROC curve of two models |
| confusion_matrix_comparison.png | Confusion matrix of two models |

## Some small problems

- We use SimHei font for Chinese text in the pictures. If your computer don't have this font, maybe the Chinese words will show as small squares. You can change the font name in code to fix it.
- This is our first machine learning project, so the code maybe not very perfect. If you have any suggestions or find any bugs, please tell me. I am still learning.
- The code will run maybe a little slow if your computer is not fast, because Random Forest take some time.

## What We learned

Through this project we learned how to use sklearn and lightgbm library, how to process data before training, and how to evaluate model performance with accuracy, AUC and confusion matrix. Hope this can help other beginners too.
