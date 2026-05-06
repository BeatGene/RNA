from utils import read_yaml
from utils import instantiate_model

if __name__ == "__main__":
    config = read_yaml("config/RNA_test.yaml")

    model = instantiate_model(config["model"], config["model_args"])


    print(model)
