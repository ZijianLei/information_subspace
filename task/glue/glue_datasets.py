    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

class glue_data():
    def __init__(self,tokenizer: AutoTokenizer,args):
        super().__init__()
        # data_args,_ = args
        model_args, data_args, training_args, _, adapter_args = args
        self.tokenizer = tokenizer
        self.data_args = data_args
        #labels
        self.is_regression = data_args.dataset_name == "stsb"
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                "/home/datasets/zjlei/datasets/glue",
                data_args.dataset_name,
                cache_dir= '/home/datasets/zjlei/datasets/glue',
                use_auth_token=True if model_args.use_auth_token else None,
            )


        # Labels
        if not self.is_regression:
            label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(label_list)
        else:
            self.num_labels = 1
         # Preprocessing the raw_datasets
        if data_args.task_name is not None:
            self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if data_args.task_name is not None and not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length) 

        # maopping the raw datasets
        raw_datasets = raw_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                self.train_dataset = train_dataset.select(range(max_train_samples))

        if training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            if data_args.task_name =='mnli':
                self.eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
                self.mmdataset = raw_datasets["validation_mismatched"]
                # self.eval_dataset.append(raw_datasets["validation_mismatched"])
            else:
                self.eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                self.eval_dataset = eval_dataset.select(range(max_eval_samples))

        if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            self.predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                self.predict_dataset = predict_dataset.select(range(max_predict_samples))


         # Get the metric function
        if data_args.task_name is not None:
            self.metric = load_metric("task/glue/glue.py", data_args.dataset_name)
    
        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
        # we already did the padding.
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            self.data_collator = None

    def compute_metrics(self,p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.task_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


    def preprocess_function(self,examples):
            # Tokenize the texts
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

            return result
