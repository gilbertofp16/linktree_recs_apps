from recommender_model.model_inference import RecommenderModelPredictorV1

recs_knn_model = RecommenderModelPredictorV1()


def test_recs_knn_model():
    recs_result = recs_knn_model.generate_user_recs("542fcccf-d465-4f53-86ca-a51453ca8c24")
    assert len(recs_result) == 6


def test_recs_knn_model_restriction_25_n():
    recs_result = recs_knn_model.generate_user_recs("542fcccf-d465-4f53-86ca-a51453ca8c24", n=100)
    assert len(recs_result) == 11


def test_recs_knn_model_not_user_from_entertainment():
    recs_result = recs_knn_model.generate_user_recs("e797096b-7ab3-4538-90ae-a140b4ebc103")
    assert len(recs_result) == 0


def test_recs_knn_model_user_empty():
    recs_result = recs_knn_model.generate_user_recs("")
    assert len(recs_result) == 0
