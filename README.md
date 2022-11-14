# LINKTREE RECS APPS

In this project we will investigate and code the recommendation system to recommend top 6 apps to an Linker.
I am only covering the vertical entertainment and USER_ID inside that vertical. 

Repo Structure:


NOTE: Before running any of the code please download the dataset from here:
https://drive.google.com/file/d/1RrlURzSB4Tb3pu8O18QI2uJZ4qt-mc0s/view?usp=share_link
Then put the folder dataset in the root of the project(linktree_recs_apps/).

Research:

In this folder we will see python notebooks with the exploratory analysis. 
To run this code you should follow the next steps:

- execute in terminal 'curl -sSL https://install.python-poetry.org |  POETRY_VERSION=1.1.15 python3 -'
- Go to research folder path and execute poetry install
- Use the new environment to run the code


Models:

In this folder we will see the training model for our recs model and then the inference module. 

train_recommender_model:
    In this folder we can see python files with code cell. I am not using notebooks because it is more complicated
    to transform this code to staging or prod if I use notebook. Still this code needs better cleaning and 
    testing. 

    To run the code inside this folder you should follow the next steps:

    - execute in terminal 'curl -sSL https://install.python-poetry.org |  POETRY_VERSION=1.1.15 python3 -'
    - Go to train_recommender_model folder path and execute poetry install
    - Use the new environment to run the code

    Then run the python files in the following order:
    1. src/training_data.py
    2. src/model_creation.py
    3. src/model_evaluation.py

recommender_inference:
    In this folder I made the inference to use the model. This is mainly a class which will give us the recommendations.
    It has basic unit testing and needs more work. But it works accordingly to the requested using the function called:
    generate_user_recs()

    To run the code inside this folder you should follow the next steps:

    - execute in terminal 'curl -sSL https://install.python-poetry.org |  POETRY_VERSION=1.1.15 python3 -'
    - Go to recommender_inference folder path and execute poetry install
    - Use the new environment to run the code

    Then test the model inference you can use the unit testing:
    1. recommender_inference/tests/unit/test_model_inference.py

    This module can be build and publish in poetry and then use the library everywhere. I avoid this step because
    lack of time, usually we use AWS artifact in my work. 