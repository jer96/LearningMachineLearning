# collaborative based systems - recommendations based on user base
# content based systems - recoommendations based on specific user
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from bookcrossing import fetch_bookcrossing_data, get_book_titles

#CHALLENGE part 1 of 3 - write your own fetch and format method for a different recommendation
#dataset. Here a good few https://gist.github.com/entaroadun/1653794 
#And take a look at the fetch_movielens method to see what it's doing 
#fetch data and format it
data = fetch_movielens(min_rating=4.0)

# new function to fetch book crossing data
book_data = fetch_bookcrossing_data(min_rating=7.0)
titles = get_book_titles()

#CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
#the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

#create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'], epochs=30, num_threads=2)

# training three new models
model_types = ['warp', 'logistic', 'warp-kos']
book_models = [LightFM(loss=model) for model in model_types]
    
#CHALLENGE part 3 of 3 - Modify this function so that it parses your dataset correctly to retrieve
#the necessary variables (products, songs, tv shows, etc.)
#then print out the recommended results 

def sample_recommendation(model, data, user_ids):

    #number of users and movies in training data
    _, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:
        
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        print_results(user_id, known_positives, top_items)

def print_results(user_id, known_positives, top_items, rec_type='Movie Lens Sample Recommendations:', model_type='warp'):
    #print out the results
    print(rec_type)
    print('Using {} model'.format(model_type))
    print("User %s" % user_id)
    print("     Known positives:")

    for x in known_positives[:3]:
        print("        %s" % x)

    print("     Recommended:")

    for x in top_items[:3]:
        print("        %s" % x)

def book_formatter(book_metadata):
    return '{} by {}'.format(book_metadata[0], book_metadata[1])


def match_titles(titles, values):
    found_titles = []
    for isbn in values:
        if(isbn in titles):
            found_titles.append(book_formatter(titles[isbn]))
    return found_titles

def sample_book_recommendation(model, data, user_ids, titles, model_type='warp', rec_type='Book Crossing Recommendations'):
    """
    new function to parse book crossing data
    """
    _, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = match_titles(titles, data['item_labels'][data['train'].tocsr()[user_id].indices])

        scores = model.predict(user_id, np.arange(n_items))

        top_items = match_titles(titles, data['item_labels'][np.argsort(-scores)])
        print_results(user_id, known_positives, top_items, rec_type, model_type)

users = [3, 25, 450]

# evaluate models 
for i,book_model in enumerate(book_models):
    book_model.fit(book_data['train'], epochs=30, num_threads=2)
    sample_book_recommendation(book_model, book_data, users, titles, model_types[i])

sample_recommendation(model, data, users)