import sys

import re
import nltk
import pickle

from sqlalchemy import create_engine
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load Data Function
    
    Inputs:
        database_filepath -> path to SQLite db
    Returns:
        X -> list all messages.
        Y -> categories for each message.
        category_names -> list all categories.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql('SELECT * FROM disaster_messages_tl', con = conn)

    X = df.message.values
    y = df.iloc[:,4:].values
    category_names = df[df.columns[4:]].columns.values
    return X, y, category_names


def tokenize(text):
    """Tokenize message (a disaster message).
    Inputs:
        text: String, A disaster message.
    Returns:
        list. contains tokens for the text.
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():

    """Build model.
    Returns:
        pipline: RandomForestClassifier. 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])
   
   # parameters = {
   # 'tfidf__use_idf': (True, False),
   #'clf__estimator__n_estimators': [10, 20]}

    #cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):

    """Evaluate model
    Inputs:
        model: RandomForestClassifier.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """

    y_pred = model.predict (X_test)
    results = pd.DataFrame(columns=['category','precision','recall','fscore', 'accuracy'])

    for i, cat in enumerate(category_names):
        precision,recall,fscore,support=precision_recall_fscore_support(y_test[:,i],y_pred[:,i], average='weighted')
        accuracy = (y_test[:,i] == y_pred[:,i]).mean()
        results = results.append({'category': cat, 'precision': precision, 'recall': recall, 'fscore' : fscore, 'accuracy': accuracy}, ignore_index = True)
    return results

def save_model(model, model_filepath):
    """Save model
    Inputs:
        model: RandomForestClassifier
        model_filepath: Trained model is saved as pickel into model_filepath
    """
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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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