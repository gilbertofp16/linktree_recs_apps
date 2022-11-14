# %%
# Feature selection and training data
import os
from statistics import mean
from typing import List, Tuple

import pandas as pd
from joblib import load
from sklearn.neighbors import NearestNeighbors

# %%
# Load training data

model_data_path = os.path.join(os.getcwd(), "model_data/")
first_dataset_training = pd.read_parquet(model_data_path + "first_dataset_training.parquet")
second_dataset_training = pd.read_parquet(model_data_path + "second_dataset_training.parquet")
# %%
# create KNN model and saved
from sklearn.neighbors import NearestNeighbors

model_save_path = os.path.join(os.getcwd(), "model/")
knn_model = load(model_save_path + "knn_rec_apps.joblib")

# %%


def transform_recommendation_to_apps(
    recommendations_list: List[int], first_dataset_training_apps_list: List[str], top_n=6
) -> List[str]:
    recommendations_list = [
        first_dataset_training_apps_list[recommendation_idx] for recommendation_idx in recommendations_list
    ]
    uniq_apps = list(set(recommendations_list))[:top_n]
    return uniq_apps


# %%
# Evaluation:
# I am using the following metrics:
# Precision at k is the proportion of recommended items in the top-k set that are relevant
# Recall at k is the proportion of relevant items found in the top-k recommendations
# https://surprise.readthedocs.io/en/latest/FAQ.html#how-to-compute-precision-k-and-recall-k


def precision_recall_at_k(
    predictions_per_linker: List[str], relevant_apps_per_linker: List[str]
) -> Tuple[List[float], List[float]]:
    """Return precision and recall at k metrics for each user"""

    precisions = []
    recalls = []
    for linker_prediction_app_list, relevant_apps_list in zip(predictions_per_linker, relevant_apps_per_linker):

        # Number of relevant items
        n_rel = len(relevant_apps_list)

        # Number of recommended items in top k
        n_rec_k = len(linker_prediction_app_list)

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = len(set(linker_prediction_app_list) & set(relevant_apps_list))

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precision_score = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        precisions.append(precision_score)

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recall_score = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        recalls.append(recall_score)

    return precisions, recalls


# %%
# I will create the evaluation data to compare performance of the baseline against our new created model
# first we will need to get the relevant items, the idea is that we saved outside of the training data linkers
# that have equal 8 apps, as hypothesis I am saying that an experimented linker
# will have 8 apps interactions or more, to get to 8 apps it needs our recommender system as guide,
# then if we take the first app that these linkers used we could recommend the next 6 apps and see
# how well our recommender does it. Also we can compare to the baseline recommendations which is:
# ["MOBILE_APP", "CLUBHOUSE", "VIDEO", "PRODUCT", "TWITTER", "HEADER"]


evaluation_data = second_dataset_training[second_dataset_training["count_per_user_app"] >= 8]

# getting first app interaction to be use as input for our model to get 6 new apps recommendations
first_app_per_linker = evaluation_data.sort_values(by="app_created_date", ascending=True).drop_duplicates(
    subset=["USER_ID"], keep="first"
)

# get all unique relevant apps that linker used, excluding first one. Everything sort by date.
linker_per_recommendations = (
    evaluation_data.sort_values(by="app_created_date", ascending=True)
    .groupby("USER_ID")["ITEM_ID"]
    .apply(list)
    .reset_index(name="all_relevant_apps")
)

linker_per_recommendations["relevant_apps"] = linker_per_recommendations["all_relevant_apps"].map(
    lambda all_relevant_apps: list(set(all_relevant_apps[1:]))
)

# finally in this step we have per linker/USER_ID the relevant apps linker used after first one and also the input
# for the model as first interaction app
linker_per_recommendations_with_first_time_app = linker_per_recommendations.merge(
    first_app_per_linker, how="left", on="USER_ID"
)

# let's take 1000 and evaluate the model and baseline
linker_per_recommendations_with_first_time_app_100 = linker_per_recommendations_with_first_time_app[:100]
# %%
# Get recommendation per linker, using as input the first interaction app as entry point. I ignore other entries to be used as
# true label for our evaluation


def get_recommendations(
    first_app_linker_list: List[str],
    knn_model: NearestNeighbors,
    first_dataset_training: pd.DataFrame,
    top_neigh_number=25,
) -> List[List[str]]:
    recommendations_list = knn_model.kneighbors(first_app_linker_list, top_neigh_number, return_distance=False)
    recommendation_list_cleaned = [
        transform_recommendation_to_apps(recommendation, list(first_dataset_training["ITEM_ID"]))
        for recommendation in recommendations_list
    ]
    return recommendation_list_cleaned


first_app_linker_list = linker_per_recommendations_with_first_time_app_100[
    ["PRODUCT_CATEGORY_mapping", "months_apps", "months_linker"]
].values.tolist()
linker_per_recommendations_with_first_time_app_100["predictions_rec_model"] = get_recommendations(
    first_app_linker_list, knn_model, first_dataset_training
)
linker_per_recommendations_with_first_time_app_100["baseline"] = [
    ["MOBILE_APP", "CLUBHOUSE", "VIDEO", "PRODUCT", "TWITTER", "HEADER"]
] * len(linker_per_recommendations_with_first_time_app_100)

# %%
predictions_per_linker = list(linker_per_recommendations_with_first_time_app_100["predictions_rec_model"])
relevant_apps_per_linker = list(linker_per_recommendations_with_first_time_app_100["relevant_apps"])
precision_per_linker, recall_per_linker = precision_recall_at_k(predictions_per_linker, relevant_apps_per_linker)

print("AVG precision using recs model", mean(precision_per_linker))
print("AVG recall using recs model", mean(recall_per_linker))


# %%
predictions_per_linker = list(linker_per_recommendations_with_first_time_app_100["baseline"])
relevant_apps_per_linker = list(linker_per_recommendations_with_first_time_app_100["relevant_apps"])
precision_per_linker, recall_per_linker = precision_recall_at_k(predictions_per_linker, relevant_apps_per_linker)

print("AVG precision using baseline", mean(precision_per_linker))
print("AVG recall using baseline", mean(recall_per_linker))
