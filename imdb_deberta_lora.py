import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import logging
import datasets
import evaluate
import torch

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split


train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    model_id = "microsoft/deberta-v2-xxlarge"

    tokenizer = AutoTokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=1024, truncation=True)
        # return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config
    )

    # print(model)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["query_proj", "key_proj", "value_proj"],
        lora_dropout=0.05,
        bias="none",
        # use_rslora=True,
        task_type=TaskType.SEQ_CLS
    )

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        # learning_rate=5e-7,

        bf16=True,
        # fp16=True,
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        # gradient_checkpointing=True,

        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,

        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_lora.csv", index=False, quoting=3)
    logging.info('result saved!')
