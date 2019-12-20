import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv('../data/disaster_messages.csv')
    categories = pd.read_csv('../data/disaster_categories.csv')
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = [name[:-2] for name in row]
    categories.columns = category_colnames
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    if os.path.exists(database_filename): os.remove(database_filename)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('diaster', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()