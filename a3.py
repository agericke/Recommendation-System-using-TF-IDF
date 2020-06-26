# coding: utf-8
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def download_data():
    """Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance|horror|HORROR'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    
    movies['tokens'] = [tokenize_string(genre) for genre in movies.genres]
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame 
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
      
    >>> download_data()
    >>> path = 'ml-latest-small'
    >>> ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    >>> movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    >>> movies = tokenize(movies)
    >>> movies = movies.head(5)
    >>> movies_short, vocab = featurize(movies)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('adventure', 0), ('animation', 1), ('children', 2), ('comedy', 3), ('drama', 4), ('fantasy', 5), ('romance', 6)]
    >>> featcol = movies_short['features'].tolist()
    >>> for feat in featcol:
    ...     feat.toarray()
    ...     type(feat)
    array([[0.39794001, 0.69897   , 0.39794001, 0.09691001, 0.        ,
            0.39794001, 0.        ]])
    <class 'scipy.sparse.csr.csr_matrix'>
    array([[0.39794001, 0.        , 0.39794001, 0.        , 0.        ,
            0.39794001, 0.        ]])
    <class 'scipy.sparse.csr.csr_matrix'>
    array([[0.        , 0.        , 0.        , 0.09691001, 0.        ,
            0.        , 0.39794001]])
    <class 'scipy.sparse.csr.csr_matrix'>
    array([[0.        , 0.        , 0.        , 0.09691001, 0.69897   ,
            0.        , 0.39794001]])
    <class 'scipy.sparse.csr.csr_matrix'>
    array([[0.        , 0.        , 0.        , 0.09691001, 0.        ,
            0.        , 0.        ]])
    <class 'scipy.sparse.csr.csr_matrix'>
    """
    
    # Vocab and df constructor
    keys_set = set()
    # First compute the number of unique documents containing each term
    #Update df with a set of the tokens just in case there are repeated elements
    df = Counter()
    N = movies.shape[0]
    for movie_tokens in movies.tokens: #.to_list(): # not needed -awc
        keys_set.update(movie_tokens)
        # We use the set statement to avoid counting several times if a document conayins an item several times
        df.update(set(movie_tokens))
    vocab = {key: index for index, key in enumerate(sorted(keys_set))}        
    
    csr_matrix_array = list()
    # For each movie, obtain the tokens and update data.
    for movie_tokens in movies.tokens: #.to_list(): # not needed -awc
        #Create secific variables for each document
        tf_counter= Counter(movie_tokens)
        tf = np.zeros(len(vocab))
        max_tokfreq_per_movie = tf_counter.most_common()[0][1]
        #Obtain max freq value for a token in the document and freq per token
        for token, index in vocab.items():
            if token in tf_counter:
                tf[index] = tf_counter[token]
        
        #Build the arrays for the csr_matrix for each document
        row_ind = []
        col_ind = []
        data = []
        row = 0
        for t in set(movie_tokens):
            if t in vocab.keys():
                row_ind.append(row)
                col_ind.append(vocab[t])
                data.append((tf[vocab[t]]/max_tokfreq_per_movie)*math.log10(N/df[t]))
            
        #Append the csr_matrix created
        csr_matrix_array.append(csr_matrix((data, (row_ind, col_ind)), shape=(1, len(vocab))))
    
    # Update datafram adding new column 'features'
    movies['features'] = csr_matrix_array
    
    return (movies, vocab)


def train_test_split(ratings):
    """
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    # use csr_matrix, not array -awc
    def norm(a):
        return math.sqrt(a.dot(a.T).sum())
    return a.dot(b.T).sum() / ((norm(a) * norm(b)))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    # Initialize numpy array
    pred_ratings = np.array([])
    # For each row, Obatin all ratings of user i and compute the weighted average
    for index, row in ratings_test.iterrows():
        #Obtain all ratings that the user has already done.
        movie_rating = 0
        user_pos_ratings_sum_weight = 0
        # For every raiting the user has done:
        # Compute the cosine_sim for that movieId
        # For every other movie that u has rated
        for j, row2 in ratings_train[ratings_train.userId==int(row.userId)].iterrows():
            # Compute weight of movie m (other rated movie by user) and movie i (The actual movie we want to predict the rating for)
            weight = cosine_sim(movies[movies.movieId==int(row2.movieId)].features.values[0], movies[movies.movieId==int(row.movieId)].features.values[0])
            if  weight > 0:
                movie_rating += weight*float(row2.rating)
                user_pos_ratings_sum_weight += weight
        
        if user_pos_ratings_sum_weight == 0:    
            movie_rating = ratings_train[ratings_train.userId==int(row.userId)].rating.mean()
            pred_ratings = np.append(pred_ratings, movie_rating)
        else:
            pred_ratings = np.append(pred_ratings, movie_rating/user_pos_ratings_sum_weight)
        
    return pred_ratings


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
