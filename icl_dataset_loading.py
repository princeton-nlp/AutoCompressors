import datasets
import pickle

def get_dataset(args):
    if args.dataset == "ag_news":
        train_dataset = datasets.load_dataset("ag_news")["train"]
        test_dataset = datasets.load_dataset("ag_news")["test"]
        options = ["World", "Sports", "Business", "Sci/Tech"]
        template = "Article: {text}\nTopic: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["text"]
        recalibrate_every = False
        balanced_sampling = False
        
    elif args.dataset == "glue/sst2" or args.dataset == "sst2":
        train_dataset = datasets.load_dataset("glue", "sst2")["train"]
        test_dataset = datasets.load_dataset("glue", "sst2")["validation"]
        options = ["negative", "positive"]
        template = "Sentence: {sentence}\nSentiment: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["sentence"]
        recalibrate_every = False
        balanced_sampling = True
        
    elif args.dataset == "super_glue/boolq" or args.dataset == "boolq":
        train_dataset = datasets.load_dataset("super_glue", "boolq")["train"]
        test_dataset = datasets.load_dataset("super_glue", "boolq")["validation"]
        options = ["incorrect", "correct"]
        template = "{passage}\nquestion: {question}?\nanswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["passage"]
        recalibrate_every = True
        balanced_sampling = False

    elif args.dataset == "super_glue/wic" or args.dataset == "wic":
        train_dataset = datasets.load_dataset("super_glue", "wic")["train"]
        test_dataset = datasets.load_dataset("super_glue", "wic")["validation"]
        options = ["no", "yes"]
        template = "{sentence1}\n{sentence2}\nquestion: Is the word '{word}' used the same way in the two sentences above?\nanswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["sentence1", "sentence2"]
        recalibrate_every = True
        balanced_sampling = False
        
    elif args.dataset == "super_glue/wsc" or args.dataset == "wsc":
        train_dataset = datasets.load_dataset("super_glue", "wsc")["train"]
        test_dataset = datasets.load_dataset("super_glue", "wsc")["validation"]
        options = ["no", "yes"]
        template = "Question: In the sentence \"{text}\", does the pronoun '{span2_text}' refer to {span1_text}?\nAnswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

        input_keys = ["text"]
        recalibrate_every = True
        balanced_sampling = False
    
    elif args.dataset == "super_glue/rte" or args.dataset == "rte":
        train_dataset = datasets.load_dataset("super_glue", "rte")["train"]
        test_dataset = datasets.load_dataset("super_glue", "rte")["validation"]
        options = ["True", "False"]
        template = "{premise}\nquestion: {hypothesis} True or False?\nanswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

        input_keys = ["hypothesis"]
        recalibrate_every = True
        balanced_sampling = False
    
    elif args.dataset == "super_glue/cb" or args.dataset == "cb":
        train_dataset = datasets.load_dataset("super_glue", "cb")["train"]
        test_dataset = datasets.load_dataset("super_glue", "cb")["validation"]
        options = ["true", "false", "neither"]
        template = "{premise}\nquestion: {hypothesis}. true, false or neither?\nanswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

        input_keys = ["premise"]
        recalibrate_every = True
        balanced_sampling = True

    elif args.dataset == "super_glue/copa" or args.dataset == "copa":
        train_dataset = datasets.load_dataset("super_glue", "copa")["validation"]
        test_dataset = datasets.load_dataset("super_glue", "copa")["train"]
        template = "Context: {premise}\nAnswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": [example["choice1"], example["choice2"]]})
        test_dataset = test_dataset.map(lambda example: {**example, "options": [example["choice1"], example["choice2"]]})

        input_keys = ["premise"]
        recalibrate_every = True
        balanced_sampling = True
    
    elif args.dataset == "super_glue/multirc" or args.dataset == "multirc":
        train_dataset = datasets.load_dataset("super_glue", "multirc")["train"]
        test_dataset = datasets.load_dataset("super_glue", "multirc")["validation"]
        options = ["incorrect", "correct"]
        template = "Context: {paragraph}\n{question}\n{answer}\nanswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["paragraph"]
        recalibrate_every = True
        balanced_sampling = True

    elif args.dataset == "subj":
        train_dataset = datasets.load_dataset("SetFit/subj")["train"]
        test_dataset = datasets.load_dataset("SetFit/subj")["test"]
        options = ["objective", "subjective"]
        template = "input: {text}\ntype: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

       	input_keys = ["text"]

        recalibrate_every = False
        balanced_sampling = True
    elif args.dataset == "mr":
       	train_dataset = datasets.load_dataset("rotten_tomatoes")["train"]
        test_dataset = datasets.load_dataset("rotten_tomatoes")["test"]
        options = ["negative", "positive"]
        template = "Review: {text}\nSentiment: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

        input_keys = ["text"]

        recalibrate_every = False
       	balanced_sampling = True
    else:
        raise NotImplementedError

    return {
        "train": train_dataset,
        "test": test_dataset,
        "template": template,
        "input_keys": input_keys,
        "recalibrate_every": recalibrate_every,
        "balanced_sampling": balanced_sampling
    }
