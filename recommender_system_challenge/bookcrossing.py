import pandas as pd 
import numpy as np 
import scipy.sparse as sp 

ROOT = './BX-CSV-Dump/{}'

def get_ratings_data():
    """
    Return raw data from book crossing csv file in the form of a pandas df
    """
    df = pd.read_csv(ROOT.format('BX-Book-Ratings.csv'), sep=';', error_bad_lines=False)
    pd.to_numeric(df['Book-Rating'])
    return df

def create_mapping(values):
    """
    Create mappings from ISBN/User-ID to integer values
    """
    return {values[i] : i for i in range(len(values))}

def get_dimensions_and_split(data, test_size=.25):
    """
    Get dimensionality of data, find unique User-IDs and ISBNs and split data
    """
    users = data['User-ID'].unique()
    books = data['ISBN'].unique()
    
    num_users = len(users)
    num_books = len(books)

    user_map = create_mapping(users)
    book_map = create_mapping(books)
    
    length = len(data)
    split = int(length * (1-test_size))
    train = data[:split]
    test = data[split:]
    
    return num_users, num_books, train, test, user_map, book_map

def create_interaction_matrix(rows, cols, data, min_rating):
    """
    Create coordinate matrix in the shape num_users * num_books
    """
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)
    for new_id, isbn, rating in data:
        if(int(rating) >= min_rating):
            mat[new_id, isbn] = rating
    return mat.tocoo()
    
def parse(data):
    for val in data:
        yield val[3], val[4], val[2]

def map_to_array(map):
    values = np.empty(len(map), dtype=np.object)
    for key in map:
        values[map[key]] = key
    return values

def fetch_bookcrossing_data(min_rating=7):
    """
    Return data dictionary containing train/test data along with book titles and user id arrays
    """
    ratings = get_ratings_data()
    num_users, num_books, train, test, user_map, book_map = get_dimensions_and_split(ratings)
    
    train['new_id'] = train['User-ID'].apply(lambda row: user_map[row])
    train['new_isbn'] = train['ISBN'].apply(lambda row: book_map[row])
    test['new_id'] = test['User-ID'].apply(lambda row: user_map[row])
    test['new_isbn'] = test['ISBN'].apply(lambda row: book_map[row])

    train_interaction_mat = create_interaction_matrix(num_users, num_books, parse(train.values), min_rating)
    test_interaction_mat = create_interaction_matrix(num_users, num_books, parse(test.values), min_rating)

    return {
        'train': train_interaction_mat,
        'test': test_interaction_mat, 
        'item_labels' : map_to_array(book_map),
        'users' : map_to_array(user_map)
    }

def get_book_titles():
    """"
    Return raw titles from other csv
    """
    df = pd.read_csv(ROOT.format('BX-Books.csv'), sep=';', error_bad_lines=False, encoding = "ISO-8859-1")
    df = df[['ISBN','Book-Title', 'Book-Author']]
    values = df.values
    titles = {}
    for value in values:
        titles[value[0]] = value[1:]
    return titles
