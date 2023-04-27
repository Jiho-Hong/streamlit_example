import transformers
import torch
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx])

    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, sentence1, sentence2):
        super().__init__()

        self.model_name = model_name

        self.sentence1 = sentence1
        self.sentence2 = sentence2

        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=160)

    def tokenizing(self):
        data = []
        text = self.sentence1 + '[SEP]' + self.sentence2
        outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
        data.append(outputs['input_ids'])
        return data

    def preprocessing(self):
        return self.tokenizing()

    def setup(self, stage="fit"):
        if stage != "fit":
            predict_inputs = self.preprocessing()

            self.predict_dataset = Dataset(predict_inputs)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, vocab_size):
        super().__init__()

        self.model_name = model_name

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)

        self.plm.resize_token_embeddings(vocab_size)

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def predict_step(self, batch, batch_idx):
        x = batch

        logits = self(x)

        return logits.squeeze()


def sts_inference(sentence1, sentence2):
    trainer = pl.Trainer(accelerator='cpu')

    model_file_path = "./model/snunlp-KR-ELECTRA-discriminator-sts-epoch=13-val_pearson=0.930.ckpt"
    model = Model.load_from_checkpoint(model_file_path)

    model_name = "snunlp/KR-ELECTRA-discriminator"

    dataloader = Dataloader(model_name, sentence1, sentence2)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    score = predictions[0].item()

    if score < 0:
        score = 0
    elif score > 5:
        score = 5

    return int(score * 20)
