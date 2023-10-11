import argparse
from glob import glob
import os
import datasets
from itertools import chain

from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast, LlamaTokenizerFast






def tokenize_function(examples, tokenizer, add_special_tokens=False):
    return tokenizer(examples["text"], add_special_tokens=add_special_tokens)


def translate_num(i="6B"):
    if i.endswith("B"):
        n = float(i[:-1]) * 1000000000
    elif i.endswith("M"):
        n = float(i[:-1]) * 1000000
    elif i.endswith("K"):
        n = float(i[:-1]) * 1000
    else:
        n = 0
    return int(n)

def group_texts_by_words(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: " ".join(examples[k]).split() for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # print("total length", total_length)

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
        result = {
        k: [" ".join(t[i : i + block_size]) for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    else:
        result = { 
            k: [" ".join(t)]
        for k,t in concatenated_examples.items()}
    return result

def group_texts(examples, block_size, make_labels=False):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # print("total length", total_length)

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
        result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    else:
        result = { 
            k: [t]
        for k,t in concatenated_examples.items()}
    if make_labels:
        result["labels"] = result["input_ids"].copy()

    return result




def preprocess(dataset, tokenizer, args):

    print(dataset)
    # dataset = dataset.map(lambda x: group_texts(x, 100*args.block_size, make_labels=False), 
    #                                     batched=True, batch_size=10, num_proc=100, desc="Grouping texts...")
    columns_to_remove = list(dataset.features.keys())
    tokenized_data = dataset.map(lambda x: tokenize_function(x, tokenizer),
                                batched=True, remove_columns=columns_to_remove, batch_size=1, num_proc=32,  desc="Tokenizing...",)
    tokenized_data = tokenized_data.map(lambda x: group_texts(x, args.block_size, make_labels=True), 
                                        batched=True, batch_size=10, num_proc=32, desc="Grouping texts...")
    print(tokenized_data)
    tokenized_data = tokenized_data.filter(lambda x: (len(x["input_ids"]) >= args.block_size), num_proc=32)
    tokenized_data = tokenized_data.train_test_split(test_size=min(args.total_validation//args.block_size, int(len(tokenized_data)*0.05)),
                                    train_size=min(args.total_train//args.block_size, int(len(tokenized_data)*0.9)))
    print(tokenized_data)
    tokenized_data.save_to_disk(os.path.join(args.out_dir, args.custom_dataset_suffix))
 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset")
parser.add_argument("--tokenizer", required=True, dest="tokenizer")
parser.add_argument("--block_size", required=True, dest="block_size", type=int)
parser.add_argument("--total_train", required=True, dest="total_train")
parser.add_argument("--total_validation", required=True, dest="total_validation")
parser.add_argument("--outdir", required=True, dest="out_dir")
parser.add_argument("--custom_dataset_suffix", required=False, default="", dest="custom_dataset_suffix")

args = parser.parse_args()

args.total_train = translate_num(args.total_train)
args.total_validation = translate_num(args.total_validation)
dataset=load_from_disk(args.dataset)

# dataset=dataset.select(range(100000))

#filter out short documents
print(dataset)
dataset = dataset.filter(lambda x : (len(x["text"].split(" ")) > args.block_size / 2), desc="Filtering out short documents...")

columns_to_remove = [c for c in list(dataset.features.keys()) if c != "text"]
dataset = dataset.remove_columns(column_names=columns_to_remove)

print(dataset.features)
#shorten documents to make tokenization faster
dataset = dataset.map(lambda x: group_texts_by_words(x, args.block_size), 
                                    batched=True, batch_size=10, num_proc=32, desc="Grouping untokenized texts...")

#restrict to initial segment of dataset with enough words
if args.dataset.split("/")[-1] not in ["wiki", "stack_exchange"]:
    text_lengths = [len(dataset[i]["text"].split(" ")) for i in range(len(dataset))]
    cumsum=0
    print("Finding cutoff")
    for n, i in enumerate(text_lengths):
        cumsum+=i
        if cumsum > (args.total_train + args.total_validation):
            break
    print(f"Restricting to the first {n} rows, out of {len(dataset)}")
    dataset = dataset.select(range(n))
else:
    print("skipping trimming")

tokenizer= LlamaTokenizerFast.from_pretrained(args.tokenizer)

tokenizer.model_max_length = 10000000000
processed_dataset = preprocess(dataset, tokenizer, args)