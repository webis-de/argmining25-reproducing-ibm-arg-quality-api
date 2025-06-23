import os
from pathlib import Path
import json
import datetime
import time
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from .utils import get_device, format_time
from .utils import set_seed
from .dataset import get_dataloader, load_data


DEVICE = get_device()





class CustomBERTModel(nn.Module):

    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.classifier.bias.data.fill_(0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)

        outputs = outputs.last_hidden_state[:, 0, :]
        # outputs = outputs[1]

        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        outputs = self.sigmoid(outputs)
        return outputs


class MyBertModel:

    def __init__(self, config, num_subset, model_path=None):

        self.type = config['model_type']
        self.config = config
        self.name = model_path.rsplit("/", 1)[1] if model_path is not None else ""

        if self.type == 'custom':
            self.model = CustomBERTModel(config['num_labels'])
            self.criterion = nn.MSELoss() if config["loss"] == "mse" else nn.BCEWithLogitsLoss()

        elif self.type == 'sequence':
            if model_path:
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=config['num_labels']
                )
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=config['num_labels']
                )

        set_seed(config['random_seed'])
        self.tokenizer, self.train_dl, self.val_dl, self.test_dl, self.test_args = get_dataloader(
            config["data"], config["datapath"], config["batch_size"], num_subset=num_subset)

        self.model.to(torch.device(DEVICE))

        self.optimizer = AdamW(
            self.model.parameters(), lr=config["learning_rate"], eps=config["epsilon"])
        total_steps = len(self.train_dl) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    def save(self, path, epoch):
        t = format_time(time.time()).split(", ")[1]
        d = datetime.datetime.today().strftime('%Y_%m_%d')
        self.name = f"bert_{self.type}_{d}_{t}_{epoch}"

        output_dir = f"{path}/{self.name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Saving model to %s" % output_dir)

        with open(os.path.join(output_dir, 'training_config.json'), 'w') as file:
            json.dump(self.config, file, indent=4)

        self.tokenizer.save_pretrained(output_dir)

        if self.type == 'custom':
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        elif self.type == 'sequence':
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)


    def load(self, path):
        if self.type == 'custom':
            self.model.load_state_dict(
                torch.load(
                    os.path.join(path, "model.pt"), weights_only=True,
                    map_location=torch.device(DEVICE)))

        # weights for sequence type get loaded in the __init__

        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.eval()

    @staticmethod
    def load_fine_tuned(path):
        with open(os.path.join(path, 'training_config.json'), 'r') as file:
            config = json.load(file)

        model = MyBertModel(config, num_subset=0, model_path=path)
        model.load(path)

        return model

    def train(self, epochs):
        print('Start Training...')
        self.model.train()

        for epoch_i in range(0, epochs):
            print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            t0 = time.time()
            total_train_loss = 0

            for step, batch in enumerate(self.train_dl):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.'.format(step, len(self.train_dl), elapsed))

                self.model.zero_grad()
                b_input_ids, b_input_mask, b_labels = tuple(b.to(DEVICE) for b in batch)

                if self.type == 'custom':
                    output = self.model(b_input_ids, b_input_mask)
                    loss = self.criterion(output.squeeze(), b_labels)
                    total_train_loss += loss.item()
                    loss.backward()
                elif self.type == 'sequence':
                    output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    total_train_loss += output.loss.item()
                    output.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_dl)
            training_time = format_time(time.time() - t0)
            print("  Train Loss: {:.2f} | Training epoch took: {}".format(avg_train_loss, training_time))

            self.validate()

            self.save(self.config["model_path"], epoch_i)

        print("\nTraining complete.")

    def validate(self):
        self.model.eval()

        t0 = time.time()
        total_val_loss = []

        for batch in self.val_dl:
            b_input_ids, b_input_mask, b_labels = tuple(b.to(DEVICE) for b in batch)

            with torch.no_grad():
                if self.type == 'custom':
                    outputs = self.model(b_input_ids, b_input_mask)
                    loss = self.criterion(outputs.flatten(), b_labels)
                    total_val_loss.append(loss.item())

                elif self.type == 'sequence':
                    output = self.model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask, labels=b_labels)
                    total_val_loss.append(output.loss.item())

        avg_val_loss = sum(total_val_loss) / len(self.val_dl)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {:.2f} | Training epoch took: {}".format(avg_val_loss, validation_time))


    def predict(self, dl):
        self.model.eval()

        predictions, true_labels = [], []
        for batch in dl:
            b_input_ids, b_input_mask, b_labels = tuple(b.to(DEVICE) for b in batch)
            with torch.no_grad():

                if self.type == 'custom':
                    outputs = self.model(b_input_ids, b_input_mask)
                    predictions += outputs.squeeze().to('cpu').tolist()
                elif self.type == 'sequence':
                    outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    predictions += outputs.logits.view(1, -1).tolist()[0]

            true_labels += b_labels.to("cpu").tolist()

        return predictions, true_labels

    def test(self):
        self.model.eval()

        predictions, true_labels = self.predict(self.test_dl)
        self.save_predictions(predictions, true_labels, self.test_args, "testset_scores")

        mse = mean_squared_error(true_labels, predictions)
        pearson = pearsonr(true_labels, predictions)
        spearman = spearmanr(true_labels, predictions)
        print(f"MSE = {mse}, Pearson = {pearson}, Spearman = {spearman}")


    def apply(self):
        set_seed(self.config['random_seed'])
        data, args = load_data(
            self.tokenizer, "test", "", self.config['argspath'], load_args=True)
        test_dl = DataLoader(data, shuffle=False, batch_size=self.config["batch_size"])

        predictions, true_labels = self.predict(test_dl)

        self.save_predictions(predictions, true_labels, args, "sentence_scores")
        mse = mean_squared_error(true_labels, predictions)
        rmse = math.sqrt(mse)
        pearson = pearsonr(true_labels, predictions)
        spearman = spearmanr(true_labels, predictions)
        print(f"MSE = {mse}, RMSE = {rmse}, Pearson = {pearson}, Spearman = {spearman}")


    def save_predictions(self, predictions, true_labels, arguments, predicted_data):
        df = pd.DataFrame([[true, pred, arg] for true, pred, arg in zip(true_labels, predictions, arguments)],
                          columns=["truth", "prediction", "argument"])
        print(f"{self.config['predictions_path']}/{predicted_data}_{self.name}.csv")
        df.to_csv(f"{self.config['predictions_path']}/{predicted_data}_{self.name}.csv", index=False)



