from utils import instantiate_logger
from utils import read_yaml

if __name__ == "__main__":
    config = read_yaml("config/RNA_test.yaml")
    logger=None
    logger = instantiate_logger(
        config.get("logger"),
        config.get("logger_args"),
        task_name=config.get("task_name"),
        debug_mode=False,
        no_logger=config.get("no_logger", False),
    )

    print(logger)
