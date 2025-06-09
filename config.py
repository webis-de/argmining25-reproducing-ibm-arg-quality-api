config = {
    "num_labels": 1,
    "epochs": 5,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "epsilon": 1e-8,
    "loss": "mse",
    "random_seed": 42,
    "data": "WA",
    "datapath": "data/arg_quality_rank_30k.csv",
    "argspath": "data/sentence_scores_ibm.csv",
    "predictions_path": "data/", #"data/predictions_server_models/", #
    "model_type": "custom",
    "model_path": "models/",
}