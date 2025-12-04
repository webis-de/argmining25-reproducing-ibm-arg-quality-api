# Reproducing the Argument Quality Prediction Model of IBM's Project Debater

This repository provides training code for the paper 'Reproducing the Argument Quality Prediction of Project Debater'.

You can download the models `WA2` and `MACE-P2` from Huggingface: [webis/argument-quality-ibm-reproduced](https://huggingface.co/webis/argument-quality-ibm-reproduced).

Install the necessary packages via `pip install -r requirements.txt`.

Download the finetuned models presented in our paper and put them in the `models` directory.

Adjust the parameters (especially the data paths) in `config.py`.

## Train model
To finetune the BERT model with the specified parameters, run `python main.py`. The model will be saved after each training epoch in the directory specified in `model_path`, together with the chosen config. This config is also loaded when applying the model on other data. After the given number of epochs, the final model is automatically evaluated on the testdata, predictions are saved in the path specified in `predictions_path`.

## Apply model
To evaluate a finetuned model on external data (defined in `argspath`), pass the model name as parameter running `python main.py --modelname=<model_name>`.

## Citation
If you use the models or the code in your research, please cite the following paper describing the retraining and evaluation process:

> Ines Zelch, Matthias Hagen, Benno Stein, and Johannes Kiesel. [Reproducing the Argument Quality Prediction of Project Debater.](https://webis.de/publications.html#zelch_2025b), In Proceedings of the _12th Workshop on Argument Mining_, July 2025.


You can use the following BibTeX entry for citation:

```bibtex
@InProceedings{zelch:2025,
    author = {Ines Zelch and Matthias Hagen and Benno Stein and Johannes Kiesel},
    booktitle = {12th Workshop on Argument Mining (ArgMining 2025) at ACL},
    doi = {10.18653/v1/2025.argmining-1.17},
    editor = {Elena Chistova, Philipp Cimiano, Shohreh Haddadan, Gabriella Lapesa, Ramon Ruiz-Dolz},
    month = jul,
    numpages = 15,
    pages = {181--188},
    title = {{Reproducing the Argument Quality Prediction of Project Debater}},
    url = {https://aclanthology.org/2025.argmining-1.17/},
    year = 2025
}
```

