import os
import re

def get_last_checkpoint_or_last_model(folder):
    """modification of get_last_checkpoint from transformer.trainer_utils.
    This function will return the main folder if it contains files of the form "pytorch_model*". The default HF function ignores those and only looks
    for "checkpoint-*" folders."""
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    _re_model = re.compile("pytorch_model" + r"*")
    content = os.listdir(folder)
    models = [
        path for path in content if _re_model.search(path) is not None
    ]
    if models != []:
        return folder
    else:
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def parse_checkpoint_step(checkpoint):
    if checkpoint.split("-")[0]!= "checkpoint":
        return -1
    else:
        try:
            return int(checkpoint.split("-")[-1])
        except:
            print(f"got checkpoint name {checkpoint}, couldn't parse step")
            return -1
