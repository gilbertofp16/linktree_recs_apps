# %%
# Feature selection and training data
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

# %%
# Getting all data available
# interactions
interactions_csv_path = os.path.join(os.getcwd(), "..", "..", "..", "dataset", "interactions/")

all_files = glob.glob(os.path.join(interactions_csv_path, "*.csv"))
interactions_linkers_apps = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
interactions_linkers_apps_cleaned = interactions_linkers_apps[interactions_linkers_apps["ITEM_ID"] != "\\\\N"]

# apps
apps_csv_path = os.path.join(os.getcwd(), "..", "..", "..", "dataset", "items/")
all_files = glob.glob(os.path.join(apps_csv_path, "*.csv"))
apps = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
apps["app_created_date"] = apps["TIMESTAMP"].map(lambda date: datetime.fromtimestamp(int(date))).copy(deep=True)

# linkers
linkers_csv_path = os.path.join(os.getcwd(), "..", "..", "..", "dataset", "users/")
all_files = glob.glob(os.path.join(linkers_csv_path, "*.csv"))
linkers = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
linkers_cleaned = linkers[linkers["TIMESTAMP"] != "\\\\N"].copy(deep=True)
linkers_cleaned["user_created_date"] = (
    linkers_cleaned["TIMESTAMP"].map(lambda date: datetime.fromtimestamp(int(date))).copy(deep=True)
)
linkers_cleaned["VERTICALS"] = (
    linkers_cleaned["VERTICALS"]
    .map(lambda verticals: verticals.split("|") if verticals is not np.nan else [])
    .copy(deep=True)
)
del linkers_cleaned["TIMESTAMP"]
# %%
# merging interactions and linkers information
interaction_per_vertical_linker = interactions_linkers_apps_cleaned.merge(
    linkers_cleaned, how="left", on="USER_ID"
).explode("VERTICALS")

# %%
# Get only Entertainment linkers
interaction_per_vertical_linker_entertainment = interaction_per_vertical_linker[
    interaction_per_vertical_linker["VERTICALS"] == "entertainment"
]
interaction_per_vertical_linker_entertainment

# %%
# Getting count of apps per linker, uniq apps per interactions
interaction_per_vertical_linker_entertainment_count = (
    interaction_per_vertical_linker_entertainment.groupby(["USER_ID", "ITEM_ID"])
    .size()
    .sort_values(ascending=False)
    .reset_index(name="count_per_guest")
    .groupby(["USER_ID"])["count_per_guest"]
    .count()
    .reset_index(name="count_per_user_app")
)
interaction_per_vertical_linker_entertainment_count
# %%
# Let's merge this count per user app to the original linker interactions

interaction_per_vertical_linker_ent_features = interaction_per_vertical_linker_entertainment.merge(
    interaction_per_vertical_linker_entertainment_count, how="left", on="USER_ID"
)
interaction_per_vertical_linker_ent_features
# %%
# Let's also add information about the app
interaction_per_vertical_linker_ent_features = interaction_per_vertical_linker_ent_features.merge(
    apps, how="left", on="ITEM_ID"
)
interaction_per_vertical_linker_ent_features
# %%
# adding how many months an app has from creation date
def get_diff_months(date: datetime, today_date: datetime) -> float:
    return today_date.month - date.month + 12 * (today_date.year - date.year)


interaction_per_vertical_linker_ent_features["months_apps"] = interaction_per_vertical_linker_ent_features[
    "app_created_date"
].map(lambda date: get_diff_months(date, datetime.today()))
interaction_per_vertical_linker_ent_features
# %%
# adding how many months an linker has from creation date

interaction_per_vertical_linker_ent_features["months_linker"] = interaction_per_vertical_linker_ent_features[
    "user_created_date"
].map(lambda date: get_diff_months(date, datetime.today()))
interaction_per_vertical_linker_ent_features
# %%
# transform/factorize product category into number or mapping.
interaction_per_vertical_linker_ent_features["PRODUCT_CATEGORY_mapping"] = pd.factorize(
    interaction_per_vertical_linker_ent_features["PRODUCT_CATEGORY"]
)[0]
interaction_per_vertical_linker_ent_features

# %%
# Select columns with useful information
interaction_per_vertical_linker_ent_features.drop_duplicates(inplace=True)
training_data = interaction_per_vertical_linker_ent_features[
    [
        "USER_ID",
        "ITEM_ID",
        "PRODUCT_CATEGORY_mapping",
        "count_per_user_app",
        "months_apps",
        "months_linker",
        "app_created_date",
    ]
]
training_data

# %%
# we will divide the data in three
# First one: portion of Experienced linkers >8 apps interactions as training data and input to our model
# Second one: Evaluation data will consist in experienced linkers and non experience linkers but > 5 apps
# Third one: non experienced linkers which have count of apps on 3 or lower
# This data is for a unsupervised model, we don't have human evaluation data. But I will evaluate
# later using linkers which have already 8 apps or more, we will cover this in the following documentation.

model_data_path = os.path.join(os.getcwd(), "model_data/")

if not os.path.exists(model_data_path):
    os.makedirs(model_data_path)

training_data.to_parquet(model_data_path + "all_training_data_entertainment.parquet")

# First data
training_data_first_random = training_data[training_data["count_per_user_app"] > 8].sample(frac=1, random_state=1)
training_data_first_random.to_parquet(model_data_path + "first_dataset_training.parquet")

# second data
training_data_second_random = training_data[
    (training_data["count_per_user_app"] >= 5) & (training_data["count_per_user_app"] <= 8)
].sample(frac=1, random_state=1)

training_data_second_random.to_parquet(model_data_path + "second_dataset_training.parquet")

# third data
training_data_third_random = training_data[training_data["count_per_user_app"] <= 3].sample(frac=1, random_state=1)
training_data_third_random.to_parquet(model_data_path + "third_dataset_training.parquet")

# %%
