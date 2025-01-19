import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from scipy.special import softmax

class SoftmaxClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('vectorizer', CountVectorizer(
                        ngram_range=(1, 2)
                    )),
                    ('selector', SelectKBest(chi2, k=1010))
                ]))
            ])),
            ('classifier', LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',  # saga is better for large datasets but we are smaller dataset so use lbfgs
                max_iter=300000,
                C=1,  # Moderate regularization
                class_weight='balanced',
                random_state=42
            ))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def main():
    # Load cleaned data    
    train_df = pd.read_excel('../Model and Dataset/clean_train.xlsx', engine= 'openpyxl')
    test_df = pd.read_excel('../Model and Dataset/clean_test.xlsx', engine= 'openpyxl')

    print("Dataset sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    
    print("\nClass distribution in training data:")
    print(train_df['Expected Operation by Developer'].value_counts(normalize=True))
    
    # train softmax classifier using train data
    classifier = SoftmaxClassifier()
    X_train = train_df['Review Comment']
    y_train = train_df['Expected Operation by Developer']
    classifier.fit(X_train, y_train)
    
    # prediction made by test data
    X_test = test_df['Review Comment']
    y_pred = classifier.predict(X_test)
    
    # print classification report
    if 'Expected Operation by Developer' in test_df.columns:
        y_test = test_df['Expected Operation by Developer']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()