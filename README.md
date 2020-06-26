# Recommendation Systems Using TF-IDF

Here we'll implement a content-based recommendation algorithm using the tf-idf approach. It will use the list of genres for a movie as the content.

The data come from the MovieLens project: [data](http://grouplens.org/datasets/movielens/)

The model will predict the rating for a specific movie based in the featurized movie ratings and characteristics from the training set. Movie ratings goes from 0 to 5.

## Summary

It is important to notice that instead of using already done functions from libraries or packages, all functions and algorithms have been coded from scratch.

We have used a __TF-IDF__ approach for vectorizing the movie reviews contents. As a measure of similarity between movie reviews, we have used a __cosine_similarity__ calculation, and as the __error__ measure we have used __mean absolute error__.

We have used data from the movie lens project. The data contains __100K__ ratings.

The obtained __mean absolute error__ over the test set has been of __0.787455__-


