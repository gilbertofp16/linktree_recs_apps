from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
from joblib import load


class RecommenderModelPipeline(ABC):
    @abstractmethod
    def generate_user_recs(self, user_id: str, n=6) -> List[str]:
        raise NotImplementedError


class RecommenderModelPredictorV1(RecommenderModelPipeline):
    def __init__(self):

        self.recommender_path = Path(__file__).parents[0]
        self.model_path = str(self.recommender_path) + "/model/knn_rec_apps.joblib"
        self.model_index_per_app_path = str(self.recommender_path) + "/model/list_apps_per_index.joblib"

        # I am loading this data, for this demo, but we should preprocess this information
        # and being send in the request instead.
        self.user_id_input_db = str(self.recommender_path) + "/model/linker_entertainment_input_data.parquet"

        self.knn_model = self._load_model()

        self.model_index_per_app = self._load_model_index_per_app()
        self.model_user_id_input = self._load_user_id_input()

    def _load_model(self):
        return load(self.model_path)

    def _load_model_index_per_app(self):
        return load(self.model_index_per_app_path)

    def _load_user_id_input(self):
        return pd.read_parquet(self.user_id_input_db)

    def _transform_recommendation_to_apps(
        self, recommendations_list: List[int], first_dataset_training_apps_list: List[str], top_n=6
    ) -> List[str]:
        recommendations_list = [
            first_dataset_training_apps_list[recommendation_idx] for recommendation_idx in recommendations_list
        ]
        uniq_apps = list(set(recommendations_list))[:top_n]
        return uniq_apps

    # getting the best 25 first because some recommendation are repeated.
    def get_recommendations(self, first_app_linker: List[str], top_neigh_number=25, number_rec=6) -> List[List[str]]:
        recommendations_list = self.knn_model.kneighbors(first_app_linker, top_neigh_number, return_distance=False)
        recommendation_list_cleaned = [
            self._transform_recommendation_to_apps(recommendation, self.model_index_per_app, top_n=number_rec)
            for recommendation in recommendations_list
        ]
        return recommendation_list_cleaned[0]

    def generate_user_recs(self, user_id: str, n=6) -> List[str]:
        user = self.model_user_id_input[self.model_user_id_input["USER_ID"] == user_id]
        if len(user) > 0:
            user_input_model = user[["PRODUCT_CATEGORY_mapping", "months_apps", "months_linker"]].values.tolist()
            return self.get_recommendations(user_input_model, number_rec=n)

        return []
