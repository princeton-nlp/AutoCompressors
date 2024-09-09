import datasets
import transformers
import math
import logging
from transformers.testing_utils import CaptureLogger
import os
from itertools import chain

logger = logging.getLogger(__name__)

def load_raw_dataset(data_args, model_args):
    if data_args.dataset_name is not None:
        # Downloading and loading)datasets.load_from_disk() a dataset from the hub.
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = datasets.load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = datasets.load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    return raw_datasets


def preprocess_datasets(raw_datasets, tokenizer, data_args, training_args):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if not training_args.add_special_tokens:
        print("Removing special tokens in tokenization")

    def tokenize_function(examples, pad=False):
        texts = []
        for text in examples[text_column_name]:
            while text.startswith("</s>"):
                text = text[len("</s>"):]
            texts.append(text)
        with CaptureLogger(tok_logger) as cl:
            if pad:
                output = tokenizer(texts, padding="max_length", truncation=True, max_length=data_args.block_size, add_special_tokens=training_args.add_special_tokens)
            else:
                output = tokenizer(texts, add_special_tokens=training_args.add_special_tokens)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = datasets.DatasetDict()
        for key in raw_datasets.keys():
            pad = False
            if training_args.line_by_line_training and ((key == "train") | (key == "validation")):
                pad = True

            tokenized_datasets[key] = raw_datasets[key].map(
                lambda x: tokenize_function(x, pad=pad),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing texts...",
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if training_args.line_by_line_training:
            for key in tokenized_datasets.keys():
                if "validation" in key and key != "validation":
                    tokenized_datasets[key] = tokenized_datasets[key].map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Grouping texts together",
                    )
                else:
                    tokenized_datasets[key] = tokenized_datasets[key].map(lambda x: {"labels": x["input_ids"].copy()},
                                                                         load_from_cache_file=not data_args.overwrite_cache,)
            lm_datasets = tokenized_datasets
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if training_args.save_logits and training_args.train_data_index is not None:
        lens = len(lm_datasets["train"])
        num_data_per_index = math.ceil(training_args.train_data_percentage * lens)
        start_index = training_args.train_data_index * num_data_per_index
        end_index = min((training_args.train_data_index + 1) * num_data_per_index, lens)
        lm_datasets["train"] = lm_datasets["train"].select(range(start_index, end_index))
        print(f"Total number of training data: {lens}")
        print(f"Training data index: {training_args.train_data_index}, start index: {start_index}, end index: {end_index}")


def load_preprocessed_datasets(data_args, model_args):
    assert data_args.preprocessed_train_datasets is not None
    assert data_args.preprocessed_validation_datasets is not None
    d = {}
    for train_file in data_args.preprocessed_train_datasets:
        name = os.path.basename(train_file).split(".")[0]
        if os.path.exists(train_file):
            data = datasets.load_from_disk(train_file)
        else:
            data = datasets.load_dataset(
                train_file,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            )
        d[f"train-{name}"] = data
        print(f"Loaded {train_file} training data, {len(data)} examples")

    for valid_file in data_args.preprocessed_validation_datasets:
        name = os.path.basename(valid_file).split(".")[0]
        if os.path.exists(train_file):
            data = datasets.load_from_disk(valid_file)
        else:
            data = datasets.load_dataset(
                valid_file,
                split="test",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            )
        d[f"validation-{name}"] = data
        print(f"Loaded {valid_file} validation data, {len(data)} examples")


    train_data = []
    for key in d.keys():
        if key.startswith("train"):
            train_data.append(d[key])
    d["train"] = datasets.concatenate_datasets(train_data)

    lm_datasets = datasets.dataset_dict.DatasetDict(d)
    return lm_datasets