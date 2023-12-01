import copy
import random
import warnings
import itertools
import argparse
from tqdm import tqdm
from typing import List

import torch
import transformers

import auto_compressor
import icl_dataset_loading

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_softprompt_demonstrations", required=False, type=int, nargs='+', default=[])
    parser.add_argument("--num_plaintext_demonstrations", required=False, type=int, default=0)
    parser.add_argument("--use_calibration", required=False, action="store_true")
    parser.add_argument("--seed", required=False, type=int, default=42)
    args = parser.parse_args()
    return args

def get_model_tokenizer_device_isac(args):
    """
    Returns a model, tokenizer, device, and is_ac flag.

    args: argparse.Namespace

    returns: model: transformers.AutoModelForCausalLM | auto_compressor.AutoCompressorModel | auto_compressor.LlamaAutoCompressorModel,
            tokenizer: transformers.PreTrainedTokenizer,
            device: torch.device,
            is_ac: bool
    """

    if "autocompressor-llama" in args.model_path.lower(): # LLaMA-2 AC
        print(f"Loading LLaMA-2 AutoCompressorModel from {args.model_path}")
        model = auto_compressor.LlamaAutoCompressorModel.from_pretrained(args.model_path)
        is_ac = True

    elif "autocompressor" in args.model_path.lower(): # OPT AC
        print(f"Loading OPT AutoCompressorModel from {args.model_path}")
        model = auto_compressor.AutoCompressorModel.from_pretrained(args.model_path)
        is_ac = True

    else: # Vanilla (LLaMA-2 or OPT)
        print(f"Loading vanilla model from {args.model_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path)
        is_ac = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(torch.bfloat16)    
    model.to(device)

    return model, tokenizer, device, is_ac

class PromptGenerator(torch.utils.data.Dataset):
    def __init__(self, dataset: dict,
        tokenizer, 
        num_plaintext_demonstrations: int, 
        num_softprompt_demonstrations: List[int], 
        seed: int,
        delimiter="\n\n", 
        content_free_string="N/A"
    ):
        """
        Initializes a PromptGenerator object.

        Properties:
        self.dataset: dict
        self.tokenizer: transformers.PreTrainedTokenizer
        self.num_plaintext_demonstrations: int
        self.num_softprompt_demonstrations: list[int]
        self.delimiter: str
        self.content_free_string: str
        self.all_softprompts_demonstrations_tokens: list[torch.Tensor]
        self.plaintext_demonstrations_tokens: torch.Tensor
        """
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_plaintext_demonstrations = num_plaintext_demonstrations # int
        self.num_softprompt_demonstrations = num_softprompt_demonstrations # list[int] 
        self.delimiter = delimiter
        self.content_free_string = content_free_string

        # prevents collisions and avoids "incremental" sampling
        random.seed(10**6 * seed + 10**3 * sum(num_softprompt_demonstrations) + num_plaintext_demonstrations) 

        # sample indices for softprompt and plaintext demonstrations
        if self.dataset["balanced_sampling"]:   # create balanced sample
            label_wise_idxs = dict()
            for label in range(len(self.dataset["test"][0]["options"])):
                label_wise_idxs[label] = [i for i, example in enumerate(self.dataset["train"]) if example["label"] == label]
                random.shuffle(label_wise_idxs[label])
            zipped_label_wise_idxs = list(zip(*label_wise_idxs.values()))
            staggered_idxs = [idx for sublist in zipped_label_wise_idxs for idx in sublist] 
            num_total_demonstrations = sum(self.num_softprompt_demonstrations) + num_plaintext_demonstrations
            if len(staggered_idxs) < num_total_demonstrations:
                # repeat the list if there aren't enough examples
                staggered_idxs = staggered_idxs * (num_total_demonstrations // len(staggered_idxs) + 1)
            start_splice_pos = random.randint(0, len(staggered_idxs) - num_total_demonstrations)
            sample_idxs = staggered_idxs[start_splice_pos:start_splice_pos + num_total_demonstrations]
        else:   # using random sampling
            sample_idxs = random.sample(range(len(self.dataset["train"])), sum(num_softprompt_demonstrations) + num_plaintext_demonstrations)

        softprompt_idxs = sample_idxs[:sum(num_softprompt_demonstrations)]
        plaintext_idxs = sample_idxs[sum(num_softprompt_demonstrations):]

        if sum(self.num_softprompt_demonstrations) > 0: # if softprompt demonstrations are needed

            # splitting all softprompt demonstrations into chunks based on num_softprompt_demonstrations
            softprompt_examples = self.dataset["train"][softprompt_idxs]
            softprompt_examples = iter([dict(zip(softprompt_examples, i)) for i in zip(*softprompt_examples.values())]) # unzip dict
            chunked_softprompt_examples = [list(itertools.islice(softprompt_examples, 0, i)) for i in num_softprompt_demonstrations] 
            
            chunked_softprompt_demonstrations_tokens = []
            chunked_softprompt_demonstration_counts = []
            add_special_tokens = True   # adds start token only to the first chunk
            for chunk in chunked_softprompt_examples:
                softprompt_demonstrations_tokens, chunked_softprompt_demonstration_count = \
                    self.get_demonstrations_tokens(chunk, add_special_tokens=add_special_tokens)
                chunked_softprompt_demonstrations_tokens.append(softprompt_demonstrations_tokens)
                chunked_softprompt_demonstration_counts.append(chunked_softprompt_demonstration_count)
                add_special_tokens = False
            self.all_softprompts_demonstrations_tokens = chunked_softprompt_demonstrations_tokens # list of torch.Tensor
            self.num_softprompt_demonstrations = chunked_softprompt_demonstration_counts # revised list of int
            
        if self.num_plaintext_demonstrations > 0: # if plaintext demonstrations are needed
            plaintext_examples = self.dataset["train"][plaintext_idxs]
            plaintext_examples = [dict(zip(plaintext_examples, i)) for i in zip(*plaintext_examples.values())] # unzip dict
            self.plaintext_demonstrations_tokens, self.num_plaintext_demonstrations = \
                self.get_demonstrations_tokens(plaintext_examples, add_special_tokens=(sum(self.num_softprompt_demonstrations) == 0))

    def get_demonstration_string(self, example: dict, label=None, include_label=True, for_calibration=False) -> str:
        """
        Returns a demonstration string for a given example.

        example: dict
        label: int
        include_label: bool
        for_calibration: bool

        returns: str
        """

        example = copy.deepcopy(example)
        example["label"] = label if label is not None else example["label"]     # override label
        example["answer"] = example["options"][example["label"]] if include_label else ""
        if for_calibration:
            for input_key in self.dataset["input_keys"]:
                example[input_key] = self.content_free_string
        
        demonstration_string = self.dataset["template"].format(**example).rstrip()
        return demonstration_string
        
    def get_demonstrations_tokens(self, examples: list, add_special_tokens: bool, max_tokens=float('inf')):
        """
        Tokenizes demonstrations and returns the tokens and the number of examples that were used to create them (constrained by max_tokens).

        examples: list of dicts
        add_special_tokens: bool
        max_tokens: int

        returns: demonstrations_tokens: torch.Tensor, num_examples: int
        """
        demonstrations_string = ""
        num_examples = 0
        
        # keep adding examples until max_tokens is reached
        for example in examples:
            demonstration_string = self.get_demonstration_string(example) + self.delimiter
            extended_demonstrations_string = demonstrations_string + demonstration_string
            extended_demonstrations_tokens = self.tokenizer.encode(extended_demonstrations_string, add_special_tokens=add_special_tokens)
            
            if len(extended_demonstrations_tokens) <= max_tokens:
                demonstrations_string = extended_demonstrations_string
                num_examples += 1       
            else:
                break
                
        demonstrations_tokens = self.tokenizer.encode(demonstrations_string, add_special_tokens=add_special_tokens, return_tensors="pt")
        return demonstrations_tokens, num_examples
    
    def get_calibration_nlls(self, example: dict, model, device, is_ac: bool, softprompt=None, plaintext_demonstrations_tokens=None):
        """
        Computes the calibration NLLs for a given example.

        example: dict
        model: transformers.AutoModelForCausalLM | auto_compressor.AutoCompressorModel | auto_compressor.LlamaAutoCompressorModel
        device: torch.device
        is_ac: bool
        softprompt: torch.Tensor
        plaintext_demonstrations_tokens: torch.Tensor

        returns: calibration_nlls: torch.Tensor
        """
        assert (sum(self.num_softprompt_demonstrations) == 0) or (softprompt is not None)
        assert (self.num_plaintext_demonstrations == 0) or (plaintext_demonstrations_tokens is not None)
        add_special_tokens = ((self.num_plaintext_demonstrations + sum(self.num_softprompt_demonstrations)) == 0)
        
        unanswered_example_string = self.get_demonstration_string(example, include_label=False, for_calibration=True)
        unanswered_example_tokens = self.tokenizer.encode(unanswered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        
        calibration_nlls = []
        for label_idx in range(len(example["options"])):
            answered_example_string = self.get_demonstration_string(example, label=label_idx, for_calibration=True)
            answered_example_tokens = self.tokenizer.encode(answered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
            option_tokens = answered_example_tokens[:,unanswered_example_tokens.shape[1]:]
            option_length = option_tokens.shape[1]
            plaintext_tokens = answered_example_tokens if plaintext_demonstrations_tokens is None else \
                torch.cat([plaintext_demonstrations_tokens, answered_example_tokens], dim=1)
            
            with torch.no_grad():
                calibration_option_logits = model.forward(plaintext_tokens, softprompt=softprompt, use_cache=False)["logits"][:,-option_length-1:-1,:] \
                    if is_ac else model.forward(plaintext_tokens, use_cache=False)["logits"][:,-option_length-1:-1,:]
                calibration_log_softmax = torch.log_softmax(calibration_option_logits, dim=-1)
                calibration_nll = -torch.mean(calibration_log_softmax.gather(dim=2, index=option_tokens.unsqueeze(-1)))
                calibration_nlls.append(calibration_nll)
        
        return torch.tensor(calibration_nlls)

    def __len__(self):
        return len(self.dataset["test"])
    
    def __getitem__(self, index: int) -> dict:
        """
        Returns a dictionary containing the following keys:
            answered_example_options: list of torch.Tensor 
            answer_options: list of torch.Tensor
            answer_idx: int
            test_example: dict

        index: int

        returns: dict
        """
        test_example = self.dataset["test"][index]
        add_special_tokens = ((self.num_plaintext_demonstrations + sum(self.num_softprompt_demonstrations)) == 0)

        unanswered_example_string = self.get_demonstration_string(test_example, include_label=False)
        unanswered_example_tokens = self.tokenizer.encode(unanswered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt")
        answered_example_options_tokens = []
        options_tokens = []

        for label_idx in range(len(test_example["options"])):
            answered_example_string = self.get_demonstration_string(test_example, label=label_idx)
            answered_example_tokens = self.tokenizer.encode(answered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt")
            option_tokens = answered_example_tokens[:,unanswered_example_tokens.shape[1]:]
            answered_example_options_tokens.append(answered_example_tokens)
            options_tokens.append(option_tokens)

        return_dict = {
            "answered_example_options": answered_example_options_tokens, # full answered demonstration alternatives
            "answer_options": options_tokens, # just the answers' alternatives
            "answer_idx": test_example["label"], # correct answer index
            "test_example": test_example # original test example
        }

        return return_dict


def main(args):
    model, tokenizer, device, is_ac = get_model_tokenizer_device_isac(args)
    dataset = icl_dataset_loading.get_dataset(args)
    use_softprompt = (sum(args.num_softprompt_demonstrations) > 0)
    use_plaintext_demonstrations = (args.num_plaintext_demonstrations > 0)

    # initialize prompt generator
    prompt_generator = PromptGenerator(
        dataset=dataset, 
        tokenizer=tokenizer, 
        num_plaintext_demonstrations=args.num_plaintext_demonstrations, 
        num_softprompt_demonstrations=args.num_softprompt_demonstrations, # list
        seed=args.seed
    )

    # create softprompt if needed
    if use_softprompt:
        softprompt = None
        for softprompt_demonstrations_tokens in prompt_generator.all_softprompts_demonstrations_tokens:
            assert softprompt_demonstrations_tokens.shape[1] <= 2048, "Softprompt too long!"
            with torch.no_grad():
                softprompt = model.forward(
                    softprompt_demonstrations_tokens.to(device), 
                    softprompt=softprompt, 
                    use_cache=False, output_softprompt=True
                )["softprompt"]
    else:
        softprompt = None

    # get plaintext demonstrations
    plaintext_demonstrations_tokens = prompt_generator.plaintext_demonstrations_tokens.to(device) \
        if use_plaintext_demonstrations else None

    if args.use_calibration and not dataset["recalibrate_every"]: 
        calibration_nlls = prompt_generator.get_calibration_nlls(
            dataset["test"][0], 
            model, device, is_ac, 
            softprompt=softprompt, 
            plaintext_demonstrations_tokens=plaintext_demonstrations_tokens
        )
    
    num_correct = 0
    num_total = 0
    skip = False # flag for skipping examples that are too long

    progress_bar = tqdm(prompt_generator, mininterval=0)
    for example in progress_bar:
        if args.use_calibration and dataset["recalibrate_every"]:
            calibration_nlls = prompt_generator.get_calibration_nlls(
                example["test_example"], 
                model, device, is_ac, 
                softprompt=softprompt, 
                plaintext_demonstrations_tokens=plaintext_demonstrations_tokens
            )
        
        conditioned_nlls = []
        # iterate over all candidate answer options
        for option_idx in range(len(example["answer_options"])):
            answered_example_tokens = example["answered_example_options"][option_idx].to(device)
            option_tokens = example["answer_options"][option_idx].to(device)
            option_length = option_tokens.shape[1]
            plaintext_tokens = answered_example_tokens if plaintext_demonstrations_tokens is None else \
                torch.cat([plaintext_demonstrations_tokens, answered_example_tokens], dim=1)
            
            if (not is_ac) and (plaintext_tokens.shape[-1] > 2048):
                warnings.warn("Input longer than 2048 tokens. Skipping example!")
                skip = True
                continue

            with torch.no_grad():
                conditioned_answer_logits = model.forward(plaintext_tokens, softprompt=softprompt, use_cache=False)["logits"][:,-option_length-1:-1,:] \
                    if is_ac else model.forward(plaintext_tokens, use_cache=False)["logits"][:,-option_length-1:-1,:]
                conditioned_log_softmax = torch.log_softmax(conditioned_answer_logits, dim=-1)
                conditioned_nll = -torch.mean(conditioned_log_softmax.gather(dim=2, index=option_tokens.unsqueeze(-1)))
                conditioned_nlls.append(conditioned_nll)
        
        if skip: 
            skip = False # reset flag
            continue

        conditioned_nlls = torch.tensor(conditioned_nlls) - calibration_nlls if args.use_calibration else torch.tensor(conditioned_nlls)
        nll_answer = torch.argmin(conditioned_nlls).item()
        num_correct += int(nll_answer == example["answer_idx"])
        num_total += 1
        progress_bar.set_postfix({"accuracy": num_correct / num_total}, refresh=False)
            
    print("Accuracy:", num_correct / num_total)

if __name__ == "__main__":
    main(read_args())

"""
Example commands:

python3 evaluate_icl.py --model_path facebook/opt-2.7b --dataset ag_news --num_plaintext_demonstrations 10 --use_calibration
python3 evaluate_icl.py --model_path princeton-nlp/AutoCompressor-2.7b-6k --dataset sst2 --num_softprompt_demonstrations 30 30 30  --seed 1 --use_calibration
python3 evaluate_icl.py --model_path princeton-nlp/AutoCompressor-Llama-2-7b-6k --dataset sst2 --num_softprompt_demonstrations 30 30 30 --num_plaintext_demonstrations 2 --seed 1 --use_calibration
python3 evaluate_icl.py --model_path princeton-nlp/FullAttention-Llama-2-7b-6k --dataset ag_news --num_plaintext_demonstrations 5 --seed 1 
"""
