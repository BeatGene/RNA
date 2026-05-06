from utils import read_yaml
from utils import instantiate_callbacks

if __name__ == "__main__":
    config = read_yaml("config/RNA_test.yaml")

    callbacks = instantiate_callbacks(config["callbacks"])


    print(callbacks)


