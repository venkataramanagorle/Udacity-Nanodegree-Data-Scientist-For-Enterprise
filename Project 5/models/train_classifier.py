import sys
import os
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import pickle

#Import Libraries for NLP
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Machine Learning Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support

#Metrics
from sklearn.metrics import precision_recall_fscore_support

'''
Returns X for train feature, Y for target and target category names
'''
def load_data(database_filepath):
    engine = create_engine('sqlite:///../data/diaster_db.db')
    df = pd.read_sql_table('diaster', con=engine)
    dummies = pd.get_dummies(df[['genre','related']].astype(str))
    X = df['message']
    Y = pd.concat([df[df.columns.drop(['id','message','original','genre','related'])], dummies], axis=1)
    return X, Y, Y.columns

'''
Function will tokenize, remove stop words and lemmatize the given text.
'''
def tokenize(text):
    #Normalize text. Replace Non Alphanumeric with space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #Tokenize
    words = word_tokenize(text)
    #Remove Stop Words
    words = [word for word in words if word not in stopwords.words('english')]
    #Lemmatize
    tokens = [WordNetLemmatizer().lemmatize(word).lower().strip() for word in words]
    return tokens

'''
Build Pipeline model
'''
def build_model():
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
           ])
    
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}
    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(Y_test, Y_pred):
    metrics = pd.DataFrame(columns=['Feature','f_score','precision','recall'])
    column_no = 0
    for feature in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[feature], Y_pred[:,column_no], average='weighted')
        metrics.set_value(column_no, 'Feature',   feature)
        metrics.set_value(column_no, 'f_score',   f_score)
        metrics.set_value(column_no, 'precision', precision)
        metrics.set_value(column_no, 'recall',    recall)
        column_no = column_no+1
    print('Average f_score:',  metrics.f_score.mean())
    print('Average precision:',metrics.precision.mean())
    print('Average recall:',   metrics.recall.mean())


def save_model(model, model_filepath):
    if os.path.exists(model_filepath): os.remove(model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #Predict with model
        Y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(Y_test, Y_pred)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()