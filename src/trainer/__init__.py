import importlib
import os

TRAINER_REGISTRY = {}

__all__ = "Trainer"


def register_trainer(name):
    def register_trainer(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate Trainer ({})".format(name))
        TRAINER_REGISTRY[name] = cls

        return cls

    return register_trainer


def import_trainer(trainer_dir, namespace):
    for file in os.listdir(trainer_dir):
        path = os.path.join(trainer_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            trainer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + trainer_name)


# automatically import any Python files in the trainer/ directory
trainer_dir = os.path.dirname(__file__)
import_trainer(trainer_dir, "src.trainer")
