import os
import json
import numpy as np
import torch
from huggingface_hub import login
from datasets import load_dataset
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from torch import nn
import pandas as pd
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Literal
from tqdm import tqdm

import numpy as np
import torch
from accelerate import PartialState
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.trainer import TrainerCallback
from transformers.trainer_utils import has_length

assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'

def construct_distill_data(path_file):
    dir = './cache/probs_unsafe_r2guard'

    if 'advbench_string' in path_file:
        files_to_merge = ['advbench_string.jsonl']
    else:
        files_to_merge = ['openaimod.jsonl', 'ours.jsonl', 'overkill.jsonl', 'toxicchat.jsonl', 'xstest.jsonl']
    files_to_merge = [os.path.join(dir, p) for p in files_to_merge]

    with open(path_file, 'w') as outfile:
        for file_path in files_to_merge:
            with open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)

# customize loss function
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def _prepare_non_packed_dataloader(
            self,
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func=None,
            add_special_tokens=True,
            remove_unused_columns=True,
    ):


        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True


            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "prob_unsafe": element["prob_unsafe"]}

        signature_columns = ["input_ids", "labels", "attention_mask", "prob_unsafe"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )


        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns= ['messages', 'label'],
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def compute_loss(self, model, inputs, return_outputs=False):


        probs_unsafe = inputs.get("prob_unsafe")

        inputs.pop("prob_unsafe")


        outputs = model(**inputs)
        logits = outputs.get("logits")


        loss_fct = nn.CrossEntropyLoss()

        logits = logits[:,:-1]
        labels = inputs['labels']
        labels = labels[:,1:]

        target_idx = (labels == 88850).nonzero(as_tuple=False)

        # Our SFT loss
        loss = None
        for i in range(len(logits)):
            if loss==None:
                loss = (logits[target_idx[i][0], target_idx[i][1]][88850].sigmoid() - probs_unsafe[i])**2
            else:
                loss += (logits[target_idx[i][0], target_idx[i][1]][88850].sigmoid() - probs_unsafe[i])**2

        # original CE loss
        # loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False):
    #
    #
    #     outputs = model(**inputs)
    #
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]
    #
    #     if isinstance(outputs, dict) and "loss" not in outputs:
    #         raise ValueError(
    #             "The model did not return a loss from the inputs, only the following keys: "
    #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #         )
    #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #
    #     print(f'loss.shape: {loss}')
    #
    #     return (loss, outputs) if return_outputs else loss


class CustomSFTTester(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTTester, self).__init__(*args, **kwargs)

    def _prepare_non_packed_dataloader(
            self,
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func=None,
            add_special_tokens=True,
            remove_unused_columns=True,
    ):

        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"],
                    "prob_unsafe": element["prob_unsafe"]}

        signature_columns = ["input_ids", "labels", "attention_mask", "prob_unsafe"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=['messages', 'label'],
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset


    def compute_loss(self, model, inputs, return_outputs=False):



        probs_unsafe = inputs.get("prob_unsafe")

        inputs.pop("prob_unsafe")


        outputs = model(**inputs)
        logits = outputs.get("logits")


        loss_fct = nn.CrossEntropyLoss()

        logits = logits[:,:-1]
        labels = inputs['labels']
        labels = labels[:,1:]

        target_idx = (labels == 88850).nonzero(as_tuple=False)

        # Our SFT loss
        loss = None
        for i in range(len(logits)):
            if loss==None:
                loss = (logits[target_idx[i][0], target_idx[i][1]][88850] - probs_unsafe[i])**2
            else:
                loss += (logits[target_idx[i][0], target_idx[i][1]][88850] - probs_unsafe[i])**2

        # original CE loss
        # loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss


def lora_ft(dataset):
    # Hugging Face model id
    model_id = "google/gemma-2b"
    tokenizer_id = "philschmid/gemma-tokenizer-chatml"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'  # to prevent warnings
    tokenizer.pad_token_id = tokenizer.unk_token_id



    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir="gemma-2b-dolly-chatml",  # directory to save and repository id
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
        remove_unused_columns=False
    )

    max_seq_length = 4096  # max sequence length for model and packing of the dataset

    # message =  [{'content': "\n xxx.", 'role': 'user'}, {'content': 'Safe', 'role': 'assistant'}]
    # x = tokenizer.apply_chat_template(message, tokenize=True)
    # print(x)



    collator = DataCollatorForCompletionOnlyLM([105776, 108], tokenizer=tokenizer, mlm=False)

    def formatting_prompts_func(examples):

        if isinstance(examples["messages"], list):
            output_texts = []
            for i in range(len(examples["messages"])):
                output_texts.append(tokenizer.apply_chat_template(examples["messages"][i], tokenize=False))
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples["messages"], tokenize=False)

    trainer = CustomSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False, #True
        data_collator=collator,
        formatting_func=formatting_prompts_func,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model
    trainer.save_model()

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

def test(dataset):
    peft_model_id = "gemma-2b-dolly-chatml"

    # Load Model with PEFT adapter
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)

    args = TrainingArguments(
        output_dir="tmp",  # directory to save and repository id
        num_train_epochs=0,  # number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        learning_rate=0.0,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
        remove_unused_columns=False
    )

    collator = DataCollatorForCompletionOnlyLM([105776, 108], tokenizer=tokenizer, mlm=False)
    tester = CustomSFTTester(model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        max_seq_length=4096,
        tokenizer=tokenizer,
        packing=False,  # True
        data_collator=collator,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        })


    dataloader = tester.get_train_dataloader()

    with torch.no_grad():
        preds = []
        probs_mal = []
        correct = 0
        for inputs in tqdm(dataloader):
            inputs = tester._prepare_inputs(inputs)

            probs_unsafe = inputs.get("prob_unsafe")
            probs_mal = probs_mal + probs_unsafe.cpu().tolist()

            inputs.pop("prob_unsafe")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            logits = logits[:, :-1]
            labels = inputs['labels']
            labels = labels[:, 1:]

            target_idx = (labels == 88850).nonzero(as_tuple=False)

            # Our SFT loss
            loss = None

            for i in range(len(logits)):
                prediction = logits[target_idx[i][0], target_idx[i][1]][88850].sigmoid().cpu()
                if prediction > 0.5:
                    correct += 1
                preds.append(prediction)


    probs_mal = np.array(probs_mal)
    preds = np.array(preds)
    loss = np.mean((probs_mal - preds)**2)
    print(f'loss: {loss}')
    print(f'Robust acc: {1.0 * correct / len(preds)}')


if __name__=='__main__':
    path_file = './data/distill_data/data.jsonl'
    # path_file = './data/distill_data/data_advbench_string.jsonl'
    construct_distill_data(path_file)

    distill_data = []
    with open(path_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            distill_data.append(entry)
    distill_data = datasets.Dataset.from_pandas(pd.DataFrame(data=distill_data))

    # unsafe_scores = []
    # for d in distill_data:
    #     unsafe_scores.append(d["prob_unsafe"])
    # unsafe_scores = np.array(unsafe_scores)
    # print(np.mean(unsafe_scores))
    # print(np.var(unsafe_scores))
    # print(np.min(unsafe_scores))

    lora_ft(distill_data)

    # test(distill_data)

