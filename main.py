import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.utils import set_seed
from src.evaluation import compare_quality_scores, find_outliers
from src.bert import MyBertModel
from config import config

set_seed(config["random_seed"])


def evaluate_model(modelname):
    model = MyBertModel.load_fine_tuned(config["model_path"] + modelname)
    model.apply()
    compare_quality_scores()
    find_outliers(config, modelname)


def train_model():
    num_subset = 0      # 0 for full dataset, otherwise number of arguments to use in training
    model = MyBertModel(config, num_subset=num_subset)
    model.train(config["epochs"])
    model.test()
    model.save(config["model_path"], config["epochs"])
    return model.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", required=False)
    args = parser.parse_args()
    modelname = args.modelname

    if modelname is None:
        print("No model given: train model")
        modelname = train_model()

    print("Evaluate model", modelname)
    evaluate_model(modelname)