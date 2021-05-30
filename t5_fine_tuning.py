# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={"base_uri": "https://localhost:8080/", "height": 121} colab_type="code" id="PJX4vkjj6wYz" outputId="83a8a420-48cd-4d49-bc60-2693268481c6"
from google.colab import drive
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/", "height": 302} colab_type="code" id="1V5cInhu42Wk" outputId="5501a5f1-fc49-4df7-f7a0-31cc37647337"
# !nvidia-smi

# + [markdown] colab_type="text" id="epWcPHhJ7v7j"
# Instal apex if you want to do 16 bit training. You'll probably need to restart the notebook after installing apex

# + colab={} colab_type="code" id="k1Xy7ZG-7gHt"
# # !export CUDA_HOME=/usr/local/cuda-10.1
# # !git clone https://github.com/NVIDIA/apex
# # !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="SDVQ04fGRb1v" outputId="11689986-ca27-4ab0-f14d-5ee4f0eba40d"
# !pip install transformers
# !pip install pytorch_lightning

# + [markdown] colab_type="text" id="HVxGfmEMCKs_"
# ## T5 fine-tuning
#
# This notebook is to showcase how to fine-tune [T5 model](https://arxiv.org/abs/1910.10683) with Huggigface's [Transformers](https://github.com/huggingface/transformers/) to solve different NLP tasks using text-2-text approach proposed in the T5 paper. For demo I chose 3 non text-2-text problems just to reiterate the fact from the paper that how widely applicable this text-2-text framework is and how it can be used for different tasks without changing the model at all.
#
# This is a rough draft so if you find any issues with this notebook or have any  questions reach out to me via [Twitter](https://twitter.com/psuraj28).
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 84} colab_type="code" id="HS8mNXq6bdxq" outputId="b0a32f10-f2ef-4d49-b433-266e8206040b"
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


# + colab={} colab_type="code" id="IswYuhWaz7QJ"
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# + [markdown] colab_type="text" id="RKNr7fgzcKpZ"
# ## Model
#
# We'll be using the awesome [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning) library for training. Most of the below code is adapted from here https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py
#
# The trainer is generic and can be used for any text-2-text task. You'll just need to change the dataset. Rest of the code will stay unchanged for all the tasks.
#
# This is the most intresting and powrfull thing about the text-2-text format. You can fine-tune the model on variety of NLP tasks by just formulating the problem in text-2-text setting. No need to change hyperparameters, learning rate, optimizer or loss function. Just plug in your dataset and you are ready to go!

# + colab={} colab_type="code" id="B7uVNBtXST5X"
class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
  
  def is_logger(self):
    return self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


# + colab={} colab_type="code" id="oh1R5C-GwMqx"
logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


# + [markdown] colab_type="text" id="a4hjvsBJ5Zk5"
# Let's define the hyperparameters and other arguments. You can overide this `dict` for specific task as needed. While in most of cases you'll only need to change the `data_dir`and `output_dir`.
#
# Here the batch size is 8 and gradient_accumulation_steps are 16 so the effective batch size is 128

# + colab={} colab_type="code" id="urduopvizqTq"
args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

# + [markdown] colab_type="text" id="vfhlYUUV2NIh"
# ## IMDB review classification

# + [markdown] colab_type="text" id="b3C13iabZvwK"
# ### Download IMDB Data

# + colab={} colab_type="code" id="7R0QdcgXuIWW"
# !wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xvf aclImdb_v1.tar.gz

# + colab={} colab_type="code" id="ni1cAK7EvXSB"
train_pos_files = glob.glob('aclImdb/train/pos/*.txt')
train_neg_files = glob.glob('aclImdb/train/neg/*.txt')

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="jEsRn5pa0v8d" outputId="6977ce56-d0b4-4d9f-8548-22003bb07eaf"
len(train_pos_files), len(train_neg_files)

# + [markdown] colab_type="text" id="5zgS8KhlaPiA"
# We will use 2000 samples from the train set for validation. Let's choose 1000 postive reviews and 1000 negative reviews for validation and save them in the val directory

# + colab={} colab_type="code" id="hLvBHcXwzXrk"
# !mkdir aclImdb/val aclImdb/val/pos aclImdb/val/neg

# + colab={} colab_type="code" id="IXZmLZ1pzjiY"
random.shuffle(train_pos_files)
random.shuffle(train_neg_files)

val_pos_files = train_pos_files[:1000]
val_neg_files = train_neg_files[:1000]

# + colab={} colab_type="code" id="5yTS2Jx40UNu"
import shutil

# + colab={} colab_type="code" id="hJnJpkdb0ZKY"
for f in val_pos_files:
  shutil.move(f,  'aclImdb/val/pos')
for f in val_neg_files:
  shutil.move(f,  'aclImdb/val/neg')

# + [markdown] colab_type="text" id="qdEgCwL7cIyi"
# ### Prepare Dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 186, "referenced_widgets": ["7d8f60bfc0a248e58028b6e8a477a5f7", "72dc1e39b931429883e68c0603797896", "cde60c5e18f04ba792fff8c2ac33f470", "c0c0df12695b4a1eacf8fa4ccc0ac62c", "72ea881ce3f445a9983d858b76dd257b", "d0f0c28a14b242f8990a547ed7f87c04", "f97741534b554be3b5cdccd45c73b317", "1e70a3dc7090487fa883e932bff395cb"]} colab_type="code" id="McQC1FotigqA" outputId="f60dbf68-32cf-44e1-9a2f-f9dba38cbbac"
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="wthd9SM74RG8" outputId="52deb6bd-19c4-4071-8bcb-254925d8e4cc"
ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')
len(ids_neg), len(ids_pos)


# + [markdown] colab_type="text" id="k5sJkyI3a723"
# All the examples are converted in the text-2-text format as shown in the paper. However I didn't use any task prefix here. The examples are encoded as follows,
# if the review is positive then the target is 'positive' else 'negative'
#
# **input**:  I went to see this
# movie with my husband, and we both
# thought the acting was terrible!"
#
# **target**: negative
#
# **input**:  Despite what others say,
# I thought this movie was funny.
#
# **target**: positive

# + [markdown] colab_type="text" id="VEYmYHKGcxEq"
# The dataset below takes care of reading the review files and processing the examples in text-2-text format.
#
# It cleans the review text by removing the html tags. It also appends the eos token `</s>` at the end of input and target as required by the T5 model 
#
# For T5 max input length is 512 and we can choose the max length for target sequence depending upon our dataset. The `T5Tokenizer` encodes both 'postive' and 'negative' as a single ids so I chose the max target length 2, extra 1 for the `</s>` token

# + colab={} colab_type="code" id="IIY0GenSb72m"
class ImdbDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.pos_file_path = os.path.join(data_dir, type_path, 'pos')
    self.neg_file_path = os.path.join(data_dir, type_path, 'neg')
    
    self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
    self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._buil_examples_from_files(self.pos_files, 'positive')
    self._buil_examples_from_files(self.neg_files, 'negative')
  
  def _buil_examples_from_files(self, files, sentiment):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for path in files:
      with open(path, 'r') as f:
        text = f.read()
      
      line = text.strip()
      line = REPLACE_NO_SPACE.sub("", line) 
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line + ' </s>'

      target = sentiment + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="gsnsKY6jemsr" outputId="98885b84-7f65-4d79-b470-619def772505"
dataset = ImdbDataset(tokenizer, 'aclImdb', 'val',  max_len=512)
len(dataset)

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="7g1gz05ccAzg" outputId="b3a263f1-8b22-46bf-9a33-f58c996d684a"
data = dataset[28]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

# + [markdown] colab_type="text" id="W4cfw8bMcNdA"
# ### Train

# + colab={} colab_type="code" id="aTvkv4rzhPjy"
# !mkdir -p t5_imdb_sentiment

# + colab={} colab_type="code" id="r5ngAP4OXFqZ"
args_dict.update({'data_dir': 'aclImdb', 'output_dir': 't5_imdb_sentiment', 'num_train_epochs':2})
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


# + [markdown] colab_type="text" id="RJt_VqzEAMUg"
# Define the `get_dataset` function to return the dataset. The model calls this function to get the train and val datasets. We are defining a dataset function so that we won't need to modify the model code at all. Redefine the function to return different dataset according to the problem. While this is not the best solution for now this works 

# + colab={} colab_type="code" id="2h2aGPgp0vOf"
def get_dataset(tokenizer, type_path, args):
  return ImdbDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


# + [markdown] colab_type="text" id="4IOQpawZA9XC"
# **Initialize model**

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["f414bac332054c7f86af89b8e50c7d73", "1d9c52a1bb8843b6b0f151571cbf30a4", "ed039b8125714030b03912fb29a93ca4", "d9b445b8b3b04569adf22429259b4954", "6c61b3c76d7045eb825172ba51b3fa63", "d11ffd1efc024c1ca86276430d29fd1e", "22fac35d924f464ca0b33be21a566a86", "cfe128b0d2c648c18d2255b3f8506a09", "c34ac6d2548249819c1eab28956edec4", "de2c77b3fb0f4dba99f92062b2db5328", "6ea23f0979824aac935f3f1ad10a86cd", "6452bc3b5ad445a8a5e272207fe4504d", "d6ef508766c54f8993d1d1f3d7cac040", "1b69bbddeb244defab9e21690a45c79e", "4a2b56fd6780470ab1574509fa432183", "3853231cd966465882a93fad9c5dc428"]} colab_type="code" id="kJsz3a4SilAF" outputId="d711c5a7-4c7d-4392-8cf5-3df1cbcf2859"
model = T5FineTuner(args)

# + [markdown] colab_type="text" id="RSJytKv1BFyc"
# **Initialize trainer**

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="PxO8OTA3irbw" outputId="6ebd7f3f-09fe-4363-9869-24d39183d2ff"
trainer = pl.Trainer(**train_params)

# + [markdown] colab_type="text" id="Wo7cSSvFGEhe"
# **start fine-tuning**

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["915a0b65612243668570c555a47a6c37", "c85b348624504af294b78de744969493", "d56a6918840e4f6588af5da5f8f54015", "41db48cf488a4522b1f04b33c2261262", "8c2d9ac8c22f486299949f4cbed16437", "222974dba69145e7b171360bec239ba5", "9e95200811bb497ab0ac0229f5e0ddaa", "3773b14f23974ad3a5bbb7ff947e68ca", "3ec26f803d124dd0877e1ce0e3517f68", "aabb0b2f2ae64684a80f1ea39c9a7d1b", "885696e0606c4353a5d21feec03aebc7", "659dd7302f3a40038834c4f1d8e59250", "6f3859c80aa945e4b4ae2aa957755b7c", "a840a738d20b4f43baf18453db53fdf0", "f7139c4e04374ffbafe6a849500c6369", "ef8f0b7c9b0c4f829e3ad59e83cbdd67", "dbe7a4854b8f420faaea8de4583fb1f0", "4d1f674483d44e559ae1de553dd1d726", "ce506c0137914e4db93b9db35154c62a", "e92a181ff64d4e0290236a91cbdb8d67", "e8f7179c238e4d2d91d456b2c07e1b3e", "e67100d71b5047158ab48ef0fd36cb99", "17f7e321de81404dabaa3e84fadce2cf", "a15e2fcc467242cb9fad5b2082a70c39", "f40c9bf16c9a473ba758a6439dce2652", "8d17a251bf1440d4aa8513ad5f15ba1d", "165319529b364183ae344a9a14f5bc52", "3d0c08f3abbe421d83f2b35583221291", "6e851577f682494c894b9afdd07b1201", "e67e9e945a9c430f9844946cd81aae3a", "34fbc6e29df046faaedd9fe3230559cb", "bbbdd81a2e8f4d68b33d698f45ccc9ae"]} colab_type="code" id="hVGd6imfizLP" outputId="cca18a5f-7900-4f58-ed74-6684b72a54e1"
trainer.fit(model)

# + colab={} colab_type="code" id="l-obOz6v70iB"
# !mkdir t5_base_imdb_sentiment

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="OQBJcrrWi2vC" outputId="a98adf77-6e23-4304-8ccc-5b13a33a2a32"
## save the model this way so next time you can load it using T5ForConditionalGeneration.from_pretrained
model.model.save_pretrained('t5_base_imdb_sentiment')

# + colab={} colab_type="code" id="XhjELPOk7-cz"
# # !cp -r t5_base_imdb_sentiment drive/My\ Drive/

# + [markdown] colab_type="text" id="brPOSAkjNP5t"
# ### Eval

# + [markdown] colab_type="text" id="_7SuVh05lDrJ"
# For inference we will use the `generate` method with greedy decoding with max length 2.

# + colab={} colab_type="code" id="25jbT49CVoXN"
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

# + [markdown] colab_type="text" id="cyriGR20lSRa"
# Let's visualize few predictions on test dataset

# + colab={} colab_type="code" id="wwJ998sMz2Ci"
dataset = ImdbDataset(tokenizer, 'aclImdb', 'test',  max_len=512)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# + colab={} colab_type="code" id="2LQtN5b90TyW"
it = iter(loader)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="TRD03teH0YMe" outputId="d43041e6-5d7d-49d5-e91a-7530c5d1d6b1"
batch = next(it)
batch["source_ids"].shape

# + colab={} colab_type="code" id="eewDktozk7GN"
outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="9vBe0UNw7cHY" outputId="2f0171ac-8d7d-41db-db31-d57bf72bc205"
for i in range(32):
    lines = textwrap.wrap("Review:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nActual sentiment: %s" % targets[i])
    print("Predicted sentiment: %s" % dec[i])
    print("=====================================================================\n")

# + [markdown] colab_type="text" id="lATfuiHYHq_1"
# Now predict on all the test dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["6aaf51cb9ad44c94b6a174a8768904f7", "51d23e1199274477a69557c74609afb2", "029f74818c6842d7a28af62032418880", "8db144e9144141779a1088c4bc000a99", "210517aede4f4cfab9120fdeb3d8361a", "df9bc2dc2b3c4fee98affdd7f5ca1ef6", "b684a47485af4cb1934d57cbb03a4f57", "942d20b134964d1d895af69938918464"]} colab_type="code" id="lvWQGLXhzHtn" outputId="c0f5490b-2ade-4795-fa3d-1f0f1746e23c"
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)

# + [markdown] colab_type="text" id="ZBxEcXeWGafd"
# Let's check if the model generates any invalid text

# + colab={} colab_type="code" id="Y_qylwYGXgwY"
for i, out in enumerate(outputs):
  if out not in ['positive', 'negative']:
    print(i, 'detected invalid prediction')

# + [markdown] colab_type="text" id="MpU_VkFGIgnw"
# This great is great! Our model hasn't generated any invalid prediction. Let's calculate accuarcy and other metrics

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="EdJcQODoOChP" outputId="22fc6852-5443-43e4-d87e-5a5266ddffd9"
metrics.accuracy_score(targets, outputs)

# + colab={"base_uri": "https://localhost:8080/", "height": 168} colab_type="code" id="YepnSgI5OKti" outputId="a2914edf-d572-4166-a886-6c0d731835e5"
print(metrics.classification_report(targets, outputs))

# + colab={} colab_type="code" id="UcZqrJELrRVw"


# + [markdown] colab_type="text" id="Dhqigmiw2hVh"
# ## Emotion classification
#
# While most of the sentiment-analysis datasets are binary with 'postive' and 'negative' sentiments, [Elvis Saravia](https://twitter.com/omarsar0)  has put together a great [dataset](https://github.com/dair-ai/emotion_dataset) for emotion recognition. The task is given some text classifiy the text into one of the following six emotions 
#
# 'sadness', 'joy', 'anger', 'fear', 'surprise', 'love'.
#
# Here's the [original notebook](https://colab.research.google.com/drive/1nwCE6b9PXIKhv2hvbqf1oZKIGkXMTi1X#scrollTo=pSzoz9InH0Ta) which trains ROBERTa model to classify the text

# + [markdown] colab_type="text" id="0B4IhzEgO21B"
# ### Download and view data

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="6eQhtsD65svj" outputId="a46f0a9a-27bb-4d10-c7a3-b45b3c894526"
# !wget https://www.dropbox.com/s/ikkqxfdbdec3fuj/test.txt
# !wget https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt
# !wget https://www.dropbox.com/s/2mzialpsgf9k5l3/val.txt

# + colab={} colab_type="code" id="yVrcVbvx74G5"
# !mkdir emotion_data
# !mv *.txt emotion_data

# + colab={} colab_type="code" id="jOpnh3Y06BGU"
train_path = "emotion_data/train.txt"
test_path = "emotion_data/test.txt"
val_path = "emotion_data/val.txt"

## emotion labels
label2int = {
  "sadness": 0,
  "joy": 1,
  "love": 2,
  "anger": 3,
  "fear": 4,
  "surprise": 5
}

# + colab={"base_uri": "https://localhost:8080/", "height": 313} colab_type="code" id="r4sDek6T8PXE" outputId="a061ba43-03d8-4fdc-b715-b6fca8d57388"
data = pd.read_csv(train_path, sep=";", header=None, names=['text', 'emotion'],
                               engine="python")
data.emotion.value_counts().plot.bar()

# + colab={"base_uri": "https://localhost:8080/", "height": 195} colab_type="code" id="EaKp3E1T8kkm" outputId="7b0fa7d2-199e-4e6e-b895-1d216a1be7b8"
train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 67} colab_type="code" id="i-Gt1WyPBL-6" outputId="5ca664c8-5a05-4e8c-a15b-b66891f3e164"
train.count()

# + colab={"base_uri": "https://localhost:8080/", "height": 186, "referenced_widgets": ["0037bb8409bb4d65ac4ebd956fd1e631", "db528e3117024014b4d281b650901cbd", "350fc08aa59849fc9fd3f3e454583a6c", "be936dd408314d0d90a22f627ca517ca", "99f56e1a8fdb4b2282fa6e17819d044e", "462bd815ddbc4687bcf7695f59919f0c", "40edb7d92c1145ee9e3bb823e4688e16", "f827cd8a6bf846c590913c5ea40e6737"]} colab_type="code" id="KybpXVl1Die5" outputId="1319d2b5-c84e-4c95-bae6-3af745326439"
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# + [markdown] colab_type="text" id="cANrUEXhO8QY"
# ### Dataset

# + [markdown] colab_type="text" id="8GsMQdqMPCN7"
# Here also we will process the examples in the same way we did above. If the label is 'love' we will ask the model to predict the text 'love'

# + [markdown] colab_type="text" id="AKh6m92eKZc4"
# Lets check how t5 encodes the following labels

# + colab={"base_uri": "https://localhost:8080/", "height": 118} colab_type="code" id="HDnMp5-fDIAc" outputId="837d1d28-2d17-4ff0-f345-64eed6949dbb"
emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]
for em in emotions:
  print(len(tokenizer.encode(em)))


# + [markdown] colab_type="text" id="i8VIZIWFOwMj"
# Here also all the labels are encoded as single ids

# + colab={} colab_type="code" id="8i8QD-3MDrWq"
class EmotionDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.path = os.path.join(data_dir, type_path + '.txt')

    self.data_column = "text"
    self.class_column = "emotion"
    self.data = pd.read_csv(self.path, sep=";", header=None, names=[self.data_column, self.class_column],
                            engine="python")
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    for idx in range(len(self.data)):
      input_, target = self.data.loc[idx, self.data_column], self.data.loc[idx, self.class_column]      
      
      input_ = input_ + ' </s>'
      target = target + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="kRz5jyl3FBkv" outputId="b3587087-efa7-400b-f3f4-ebc958deb33d"
dataset = EmotionDataset(tokenizer, 'emotion_data', 'val', 512)
len(dataset)

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="jxT6QzUAFQN0" outputId="68122a3a-bf3e-4125-f768-a6410abed5a9"
data = dataset[42]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

# + colab={} colab_type="code" id="PBVHtdIuFpID"


# + [markdown] colab_type="text" id="DEWi6c-pGZV9"
# ### Train

# + [markdown] colab_type="text" id="wGrpDJnLPQ0Q"
# As I said above there's no need to change the model or add task specific head or any other hyperparameters, we'll just change the dataset and that's it!

# + colab={} colab_type="code" id="kDep-uIcGYX2"
# !mkdir -p t5_emotion

# + colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" id="TgNOy7a4LJ9h" outputId="3945df44-55d0-40d2-d98c-fa196bb9d554"
args_dict.update({'data_dir': 'emotion_data', 'output_dir': 't5_emotion', 'num_train_epochs':2})
args = argparse.Namespace(**args_dict)
print(args_dict)

# + colab={} colab_type="code" id="at783kr7KvS4"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


# + colab={} colab_type="code" id="1LBvpP01KvTA"
def get_dataset(tokenizer, type_path, args):
  return EmotionDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


# + colab={"base_uri": "https://localhost:8080/", "height": 978} colab_type="code" id="v3Tty_OHGlvR" outputId="0423fedb-7a93-4990-c6ce-545b52b86e63"
model = T5FineTuner(args)

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="mIsW9pwEG27D" outputId="d0469592-9403-4397-c8cf-b2b4c48ba614"
trainer = pl.Trainer(**train_params)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["61d58772a6a64c5c8ad30dab2563a56f", "4000e73e6d804763986dc9a9c74456aa", "0dd99276ab294c939d83320f4674d5c2", "d306f7ff1ec94561aeed9ff59ba9b54b", "0893a9730450433fa76a74b008a6f482", "f8873c7201e1410cb0ec52cb7e34c3c9", "234eb8b041c44358b2f993c2853162f7", "8f73da698e85474fbecfd91bb7770c56", "26a0cb124049417aa9dbdd010e3af03a", "8a14bd8f2a424b15b48426fd5e320678", "09ed6242c5ef4a4791a1074ff7e4616e", "487a6ea92fe0463ebbcb63094fde5136", "c050be8414044acdb1a496495d148302", "56a67d534f284df0bc1121f1e264f5e2", "f168c4ae2d014e89bacc58e43427302e", "5cabe7d5ed6b46be882c558d28a29ca2", "1681a9ce7f9340caa50c4204777a6f9e", "a9f0c66f958e493286155c8d2631d255", "e04d6312d5d4425ab726588c485e668c", "fab8ee7d5d3940819eb9131efbbad791", "6dd2781f88eb4549b4203dfec9c1a98e", "893ba880ac6545baa6eb4a532ecc5753", "d4fc7ae628c94a758ce694318bc620ba", "4c33ca548b5e4738abdac09575e2a325", "ff475d6cdc074c14aa7b2cfede771b07", "d77faf8b9ea6480abe594114823ca52f", "ee4f41b591fe41a5a2d915c343b16c1d", "d8946214acc44c4cb97688538daaa33f", "9b9306452732495cbb1acd3e2fcf3b69", "f42e9e596ad0485b842fee92d1884750", "1d9f8718ba4d4b60997757ea7f1db72b", "63db466ae63b42a5a79d051ef5af653e"]} colab_type="code" id="xmk4GsEMHTfZ" outputId="ba492b59-fc67-4fd3-d42a-5965600679df"
trainer.fit(model)

# + [markdown] colab_type="text" id="GwdWdHG0RP5J"
# ### Eval

# + colab={} colab_type="code" id="dq7cCiOPRQzs"
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

# + colab={} colab_type="code" id="XKsHzqGMRQzz"
dataset = EmotionDataset(tokenizer, 'emotion_data', 'test', 512)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# + colab={} colab_type="code" id="QK7s7IpERQz5"
it = iter(loader)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="5_79Jk36RQz-" outputId="a49604ae-31da-49bc-9a90-bb5bd1366ebf"
batch = next(it)
batch["source_ids"].shape

# + colab={} colab_type="code" id="RQZKyEaVRQ0B"
outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="aAjhiBcrRQ0E" outputId="93cdd40b-310f-458d-e5ae-21debf158a39"
for i in range(32):
    c = texts[i]
    lines = textwrap.wrap("text:\n%s\n" % c, width=100)
    print("\n".join(lines))
    print("\nActual sentiment: %s" % targets[i])
    print("predicted sentiment: %s" % dec[i])
    print("=====================================================================\n")

# + [markdown] colab_type="text" id="iq8M8nbTSJlE"
# #### Test Metrics

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["8933ab7f935e4776970ddfe35f5da135", "84eb2bf17a9048fc94b6f47867d1b0ba", "cdd7554792cf4c73922e2f050d1fcaaf", "a32aa193a82f478387c14f384c2c689e", "e4cbd76c110541cbbf1386e299c4d9d6", "da67548f1abc4727965f72b8cb367681", "63b11aa7ee0c4271aedb87ad3e7d23c3", "720b90b3f86c4e5da15447777806e9a7"]} colab_type="code" id="S-oIXmoCR6kl" outputId="98bdff55-aa82-45a3-dc13-be0e78e52ea9"
dataset = EmotionDataset(tokenizer, 'emotion_data', 'test', 512)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)

# + colab={} colab_type="code" id="C9CYCGM6SRzb"
for i, out in enumerate(outputs):
  if out not in emotions:
    print(i, 'detected invalid prediction')

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="iE0WX_GbSRzq" outputId="24a4fe9c-3396-4364-aad3-8da50d456618"
metrics.accuracy_score(targets, outputs)

# + colab={"base_uri": "https://localhost:8080/", "height": 235} colab_type="code" id="mWkOZ7BASRz5" outputId="01a97ad3-3c70-43b6-e6a4-55ea5ccfa010"
print(metrics.classification_report(targets, outputs))

# + [markdown] colab_type="text" id="W6p9MGb6lWL5"
# Now lets plot  the confusion matrix and see for which classes our model is getting confused

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="9RtgfuzucFeN" outputId="0dc41da4-f99e-4469-8d0c-f055d4a18a8d"
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# + colab={} colab_type="code" id="2ioVvq5rcHZE"
cm = metrics.confusion_matrix(targets, outputs)

# + colab={"base_uri": "https://localhost:8080/", "height": 462} colab_type="code" id="4rM5XS09SSdm" outputId="171788f5-4c43-485c-b84a-133ad78e2486"
df_cm = pd.DataFrame(cm, index = ["anger", "fear", "joy", "love", "sadness", "surprise"], columns = ["anger", "fear", "joy", "love", "sadness", "surprise"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Purples', fmt='g')

# + [markdown] colab_type="text" id="QKh_bJxtlhkW"
# From the above plot we can see that the most confused classes are 'joy' and 'love' which seems obivous as these two emotions are really close. We can say the same thing 'surprise' and 'anger' as well. So our model is doing pretty well.

# + colab={} colab_type="code" id="16TiclmeX1xE"


# + [markdown] colab_type="text" id="vZ-YLmJyg64T"
# ## SWAG
#
# Now lets try a more challenging task and see how it performs.
#
# SWAG is a natural language inference and commonsense reasoning task proposed in this [paper](https://arxiv.org/pdf/1808.05326.pdf).
#
# The basic task is that  a model is
# given a context **c = (s, n)**: a complete sentence
# **s** and a noun phrase **n** that begins a second sentence, as well as a list of possible verb phrase sentence endings **V**. The model must then
# select the most appropriate verb phrase **v** in **V**. For example
#
# On stage, a woman takes a seat at the piano. She
#
# a) sits on a bench as her sister plays with the doll.
#
# b) smiles with someone as the music plays.
#
# c) is in the crowd, watching the dancers.
#
# **d) nervously sets her fingers on the keys.**
#
# The correct answer is bolded. Given the above example the model should select **nervously sets her fingers on the keys** as the most appropriate verb phrase
#
# To frame this task in text-2-text setting the example is processed as below.
#
# context: context_text options: 1: option_1 2: option_2 3: option_3 4: option_4
#
# and if the actual label is 1 then the model is asked to predict the text '1'. Here's how the above example will be processed
#
# **Input**
#
# context: On stage, a woman takes a seat at the piano. She  options: 1: sits on a bench as her sister plays with the doll. 2: smiles with someone as the music plays. 3: is in the crowd, watching the dancers. 4: nervously sets her fingers on the keys.
#
# **Target**
#
# 4
#
# This is just one possible way to process these examples, there are various other ways we can formulate this problem in text-2-text setting but that's for later.

# + [markdown] colab_type="text" id="hOxk-ZoJmamm"
# ### Dataset

# + colab={} colab_type="code" id="yeHfgOhThLPj"
import csv
from dataclasses import dataclass

from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizer


# + colab={"base_uri": "https://localhost:8080/", "height": 386} colab_type="code" id="3DulV7U5hik7" outputId="880c611b-d11c-4620-9d75-0bcfa423c1ff"
# !wget https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv
# !wget https://raw.githubusercontent.com/rowanz/swagaf/master/data/val.csv

# !mkdir swag_data
# !mv *.csv swag_data

# + colab={} colab_type="code" id="Tllm6irZg8IO"
# below code is adapted from https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/utils_multiple_choice.py

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    context: str
    endings: List[str]
    label: Optional[str]

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                # common beginning of each
                # choice is stored in "sent2".
                context=line[3],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


# + colab={} colab_type="code" id="-OXxGvqZjC9L"
class SwagDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.data_dir = data_dir
    self.type_path = type_path
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self.proc = SwagProcessor()

    self._build()
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def __len__(self):
    return len(self.inputs)
  
  def _build(self):
    if self.type_path == 'train':
      examples = self.proc.get_train_examples(self.data_dir)
    else:
      examples = self.proc.get_dev_examples(self.data_dir)
    
    for example in examples:
      self._create_features(example)
  
  def _create_features(self, example):
    input_ = example.context
    options = ['%s: %s' % (i, option) for i, option in zip('1234', example.endings)]
    options = " ".join(options)
    input_ = "context: %s  options: %s </s>" % (input_, options)
    target = "%s </s>" % str(int(example.label) + 1)

    # tokenize inputs
    tokenized_inputs = self.tokenizer.batch_encode_plus(
        [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
    )
    # tokenize targets
    tokenized_targets = self.tokenizer.batch_encode_plus(
        [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
    )

    self.inputs.append(tokenized_inputs)
    self.targets.append(tokenized_targets)


# + colab={"base_uri": "https://localhost:8080/", "height": 186, "referenced_widgets": ["78b1b91a08214461b74fb1e143247d1e", "902a509471004d2691d807c4990fccd2", "74ec15497e1743a4af6be12e3bc1487d", "a70b457d9379403f9fac247de68bb8e3", "28f9d9aa0ece4831b0f9e412d8a88f8d", "7640680e1006492da75d873726567fed", "1090e3e017564a2281c60fb53a901c75", "9df2679ba627444e9b76bd2ff0ddc657"]} colab_type="code" id="oKqFMTku3sDC" outputId="97ce9f8a-4b75-4d95-ba04-fae101f8db82"
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="PIUiU7zSpbb3" outputId="328b5f15-fe96-43ce-99e9-5d4233a7e97a"
dataset = SwagDataset(tokenizer, data_dir='swag_data', type_path='val')
len(dataset)

# + colab={"base_uri": "https://localhost:8080/", "height": 70} colab_type="code" id="zxXGbCzB37HG" outputId="8fbda79c-7be7-4d5f-8d5f-7b986a1c374b"
data = dataset[69]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

# + [markdown] colab_type="text" id="aVfmE4O3Ku7H"
# ### Train

# + colab={} colab_type="code" id="DDPxWUY86llx"
# !mkdir -p t5_swag

# + colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" id="PrWtMjcj6lmA" outputId="fe4e58ab-6916-45f9-f742-797d87ad1ef4"
args_dict.update({'data_dir': 'swag_data', 'output_dir': 't5_swag', 'num_train_epochs': 3})
args = argparse.Namespace(**args_dict)
print(args_dict)

# + colab={} colab_type="code" id="2Ojz3THj6lmK"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


# + colab={} colab_type="code" id="Kk0x0Nql6lmQ"
def get_dataset(tokenizer, type_path, args):
  return SwagDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["5c7427d7db844b9691d30cf2de1efc17", "bb0df1833ee3489da5c2a9c7b1306cc6", "3d2817812b6f475a8c838fd14646469a", "9d0f0c946790477fb8bc8bac64dfd7de", "8254b8062d5e4280bea46f8bc444c5db", "ab5f07ab5c574148a0062eb7f1ce5bcd", "47fdc2009efc443392ecd182996fcca9", "9b705e83fea84cbf912e33d6342be721", "e8e8ea6199df43019930ac7b557c46a5", "0566f29b017f47f399d7579d7929e046", "932309f0a40b46659c0cac7cc37fdc05", "da3665141bd44a24a5b5c9f36d4a9c52", "5c98e3a5b6a6403a936a725f4c30cdd3", "8da2b560fa9348098a2a7f09967d5f5f", "7e37cac227014717987922341f8099fe", "b95f98f98a76434591f90d41b43e39ba"]} colab_type="code" id="XDFGzzpQ6lmU" outputId="94aa8d13-9d11-4fa9-979f-e3bbf15bb639"
model = T5FineTuner(args)

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="1sQVILFo63Eb" outputId="57300f1a-14a8-4e26-8dac-9238e34741c0"
trainer = pl.Trainer(**train_params)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["8e79d03deee94b299431330441bd64c8", "510043ffee634f86b89ec3fc060a74ea", "e86c5fbd48ce4215a0df353122183982", "bfc3a5a3cf2e49868053db6f1ef7785d", "361a2f79ed89495894d0b09a709f8f32", "f7e53d55f0234627a3b9f2c90eb8682f", "3584c01b0c5e47dfa373bae29461e94a", "cfd9db6f31474a8189e741bf8fdad6a9", "68705cee3df5458fb5145046337d925c", "4cf1613d58bd450780ac95c994686985", "3ee5f7cf56394175900ebb14ae0b5f9e", "9f054dcf926c45459b7aa728493571a0", "b52599dda9d94c83891d1c42c5f557e0", "a1cf907a3bcc4177b1d5dd9edbf30c20", "82b29ceeb21c417782e9e29a81eb47ea", "886260804ffd4e11bc93fb6e098111ab", "69f6eb1cb0434128961b5d83529813c5", "6723d50588a248d0ad7bb118de8c3fd5", "86d71b8233c14252a897ffa29ea6d9df", "d01c708e22ab423896271fa79860e7c3", "0e8da5995754472fac5fba1f8b30d107", "3dbee77f299f4e14a1698b60d609b8a1", "8c4c9025aaae44148591ae6f8bb37347", "29e2f2f0914e4dea8117844675b42be5", "0cfc8fa73f164b4fa5ddcbc3f115ef9b", "4559bd35b33f4804b968debaaf316463", "e403cc7718bf48f1b95150482e083f02", "f6248a9db7f2466a9ab3a4fbd214f265", "475e5353d31147d3ab156c0e7835684c", "c3f65d683c6e4fe18e31ecc305f8d455", "9b50abad66b44022aa389bc3f312db6b", "762b2941ff3e47d89b6e6ce4350bc058"]} colab_type="code" id="STkqK5nC64YP" outputId="cb613d72-009f-44eb-acd8-b9c3dd44b0cb"
trainer.fit(model)

# + colab={} colab_type="code" id="o1ZB_6SK7V-3"


# + [markdown] colab_type="text" id="AgNV3TMzqSvj"
# ### Eval

# + colab={} colab_type="code" id="gFFOwfXyqc4_"
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="rsYCq3Lwqc5Y" outputId="51f7bd88-2441-42be-e8f3-adc0337a164c"
dataset =  SwagDataset(tokenizer, data_dir='swag_data', type_path='val')
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["1597779d89464892885045be715890a8", "8a42468ed6b945e8bfce1803f3ea4452", "f87eae824cf1492b9555b78648a9f261", "6cd0d574b5fd43588b8d492674125218", "17b25142ac744ba882e2bbd1f42c1db2", "09185d325ef84c1fad7b07fbd9eeed31", "ba31765789dc46229493674dab21921d", "a9dd88fb73374e108482b80993b998eb"]} colab_type="code" id="KHwMBQNjqc5h" outputId="81e7d67d-1d15-4dea-a552-695cfe8ef105"
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)

# + colab={} colab_type="code" id="ZbTValmYq15r"
for i, out in enumerate(outputs):
  if out not in "1234":
    print(i, 'detected invalid prediction')

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="jN35n2pas-pF" outputId="be8a3507-8e66-479d-c41c-dd9cb0603742"
metrics.accuracy_score(targets, outputs)

# + [markdown] colab_type="text" id="t_WaMutznvGb"
# This is great! We have achieved almost 74% accuracy with this simple formulation. This is great becuase with BERT like models to make a prediction on single example the model needs to do 4 forward passes, one for each possible endings and then the logits are concatenated together for all 4 passes and then passed through final softmax layer to produce 4 probabilities. This approach needs only a single pass for one example.

# + colab={} colab_type="code" id="rFgOHlW_tHPd"

