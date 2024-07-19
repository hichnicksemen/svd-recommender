import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

class CollaborativeRecommender:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.user_item_matrix = None
        self.pred_matrix = None

    def fit(self, ratings_df):
        """
        Fit the recommender system on the provided ratings dataframe.
        :param ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        self.user_item_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        matrix = self.user_item_matrix.values
        matrix_svd = self.svd.fit_transform(matrix)
        self.pred_matrix = np.dot(matrix_svd, self.svd.components_)

    def predict(self):
        """
        Predict the ratings for the user-item matrix.
        :return: DataFrame with predicted ratings.
        """
        if self.pred_matrix is None:
            raise ValueError("The model has not been fit yet. Please call fit() first.")
        pred_df = pd.DataFrame(self.pred_matrix, columns=self.user_item_matrix.columns, index=self.user_item_matrix.index)
        return pred_df
