# Argument_Quality_Retraining

This repository provides training code for the paper 'Reproducing the Argument Quality Prediction of Project Debater'.

You can download the models `WA2` and `MACE-P2` from (TODO).

Install the necessary packages via `pip install -r requirements.txt`.

Download the finetuned models presented in our paper and put them in the `models` directory.

Adjust the parameters (especially the data paths) in `config.py`.

## Train model
To finetune the BERT model with the specified parameters, run `python main.py`. The model will be saved after each training epoch in the directory specified in `model_path`, together with the chosen config. This config is also loaded when applying the model on other data. After the given number of epochs, the final model is automatically evaluated on the testdata, predictions are saved in the path specified in `predictions_path`.

## Apply model
To evaluate a finetuned model on external data (defined in `argspath`), pass the model name as parameter running `python main.py --modelname=<model_name>`.

If you use this model, please cite our paper:

TODO

