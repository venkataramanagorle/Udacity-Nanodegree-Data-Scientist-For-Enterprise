import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


#Import Libraries for NLP
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/disaster_db.db')
df = pd.read_sql_table('diaster', engine)

# load model
model = joblib.load("../models/model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    print(genre_counts.values)
    genre_names = list(genre_counts.index)
    
    # Top five categories count
    top_category_count = df.iloc[:,4:].sum().nlargest(5)
    top_category_names = list(top_category_count.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top Five Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()