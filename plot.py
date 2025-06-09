import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import config


def create_plot(ibm_preds, my_preds, scoretype, plotname):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect('equal')

    ax.scatter(ibm_preds, my_preds, label='Arguments')

    m_coeffs = np.polyfit(ibm_preds, my_preds, 1) # 1 for linear regression
    m_func = np.poly1d(m_coeffs)
    x_fit = np.linspace(min(ibm_preds), max(ibm_preds), 100)
    my_fit = m_func(x_fit)
    print(m_coeffs)

    ax.plot(x_fit, my_fit, label=f'Linear Regression: $y={round(m_coeffs[0], 2)}\cdot x+{round(m_coeffs[1], 2)}$', color='red', linewidth=2.5)
    ax.plot(x_fit, x_fit, label=f'Reference: $y=x$', color='orange', linestyle='--', linewidth=2.5)
    ax.set_ylabel(f'Predictions Retrained {scoretype}')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Original IBM Predictions')
    ax.grid()
    ax.legend()

    ax.set_title(f"Correlation of Predictions ({scoretype})",  fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # plt.savefig(plotname)
    plt.show()



models = ["bert_wa2", "bert_macep2"]

for model, scoretype in zip(models, ["WA", "MACE-P"]):
    preds_path = config["predictions_path"] + f"sentence_scores_{model}.csv"
    df = pd.read_csv(preds_path)
    ibm_preds = np.array( list(df["truth"]) )
    my_preds = np.array( list(df["prediction"]) )
    create_plot(ibm_preds=ibm_preds, my_preds=my_preds, scoretype=scoretype,
                plotname=f"data/plots/predictions-{scoretype}.png")

