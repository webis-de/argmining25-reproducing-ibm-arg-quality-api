import pandas as pd
import numpy as np
import heapq


def compare_quality_scores(config, modelname):
    '''
    Compare predictions of original IBM model and reproduced model:
    - calculate overall RMSE
    - count the number of predictions of the reproduced model that are lower/higher than IBM predictions
    - calculate the average difference between these lower/higher predictions compared to IBM predictions
    - we consider the original IBM predictions as ground truth (gt), the reproduced models' predictions as predictions (preds)
    '''
    df = pd.read_csv(config["predictions_path"] + f"sentence_scores_{modelname}.csv")
    gt = list(df["truth"])
    preds = list(df["prediction"])

    diff = [abs(x - y) for x, y in zip(gt, preds)]
    lower = [x-y for x, y in zip(gt, preds) if x > y]   # lower prediction than truth
    higher = [y-x for x, y in zip(gt, preds) if y > x]  # higher prediction than truth
    rmse = np.sqrt(np.mean((np.array(gt) - np.array(preds)) ** 2))

    print(f"avg difference for {modelname} is {round(np.nanmean(diff), 2)} (rmse = {rmse:.4f});" \
          f" - {len(lower)} predictions lower (avg dist. {round(np.nanmean(lower), 2)})" \
          f" - {len(higher)} predictions higher (avg dist. {round(np.nanmean(higher), 2)}))")
    

def find_outliers(config, modelname):
    filepath = config["predictions_path"] + f"sentence_scores_{modelname}.csv"
    df = pd.read_csv(filepath)
    gt = list(df["truth"])
    preds = list(df["prediction"])

    diff = [abs(x - y) for x, y in zip(gt, preds)]
    df["difference"] = diff
    df["truth"] = [round(val, 2) for val in list(df["truth"])]
    df["prediction"] = [round(val, 2) for val in list(df["prediction"])]
    
    sub001 = df[df["difference"] <= 0.001]
    sub01 = df[df["difference"] < 0.01]
    sub05 = df[df["difference"] < 0.05]
    sub10 = df[df["difference"] > 0.10]
    sub15 = df[df["difference"] > 0.15]
    sub20 = df[df["difference"] > 0.20]
    print("all:", len(df.index), " - > 0.1:", len(sub10.index), " - > 0.1:", len(sub15.index),
          " - > 0.1:", len(sub20.index), " - < 0.05:", len(sub05.index), " - < 0.01:", len(sub01.index), " - <= 0.001:", len(sub001.index))
    
    sub001.sort_values("difference").to_csv(filepath.replace(".csv", "_diff<=001.csv"), index=False)
    sub10.sort_values("difference", ascending=False).to_csv(filepath.replace(".csv", "_diff>010.csv"), index=False)
    sub15.sort_values("difference", ascending=False).to_csv(filepath.replace(".csv", "_diff>015.csv"), index=False)
    sub20.sort_values("difference", ascending=False).to_csv(filepath.replace(".csv", "_diff>020.csv"), index=False)