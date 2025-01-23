import os

def get_first_item(dataloader):
    return next(iter(dataloader))

def get_latest_model(config):
    return max([int(e.split(".")[0][4:]) for e in os.listdir(config["model_dir"]) if e.endswith(".npz") and e[0] == "a"])