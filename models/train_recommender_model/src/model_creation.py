# %%
# Feature selection and training data
import os
from statistics import mean
from typing import List, Tuple

import pandas as pd
from joblib import dump, load

# %%
# Load training data

model_data_path = os.path.join(os.getcwd(), "model_data/")
first_dataset_training = pd.read_parquet(model_data_path + "first_dataset_training.parquet")
second_dataset_training = pd.read_parquet(model_data_path + "second_dataset_training.parquet")
third_dataset_training = pd.read_parquet(model_data_path + "third_dataset_training.parquet")
# %%
# create KNN model and saved
from sklearn.neighbors import NearestNeighbors

top_neigh_number = 25

training_knn = first_dataset_training[["PRODUCT_CATEGORY_mapping", "months_apps", "months_linker"]].values.tolist()

test_user_input = third_dataset_training[["PRODUCT_CATEGORY_mapping", "months_apps", "months_linker"]].values.tolist()[
    16
]
neigh = NearestNeighbors(n_neighbors=top_neigh_number)
neigh.fit(training_knn)

model_save_path = os.path.join(os.getcwd(), "model/")

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

dump(neigh, model_save_path + "knn_rec_apps.joblib")
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


recommendations_list = knn_model.kneighbors([test_user_input], top_neigh_number, return_distance=False)
print(
    "Top 6 recommendations",
    transform_recommendation_to_apps(recommendations_list[0], list(first_dataset_training["ITEM_ID"])),
)
print("Linker previous app interaction", third_dataset_training.iloc[0])


# In this example we got related apps to the user preview apps, but we have some problem
# The first one is we should recommend unique apps and the second is the we should
# not recommend the same apps that linker is using. But this information is missing, I don't have it
# so for now I will just omitted and recommend the same app even if the linker already used before.
# %%
# Let's save also the indexes for our training data, because the KNN will give us the index and not the item id.
dump(list(first_dataset_training["ITEM_ID"]), model_save_path + "list_apps_per_index.joblib")
# %%
# Also we should save the first app that every linker used as a following process for our inference process

first_dataset_training = pd.read_parquet(model_data_path + "all_training_data_entertainment.parquet")
# getting first app interaction to be use as input for our model to get 6 new apps recommendations
first_app_per_linker = first_dataset_training.sort_values(by="app_created_date", ascending=True).drop_duplicates(
    subset=["USER_ID"], keep="first"
)
first_app_per_linker[["USER_ID","PRODUCT_CATEGORY_mapping", "months_apps", "months_linker"]].to_parquet(model_save_path +  "linker_entertainment_input_data.parquet")
# %%
