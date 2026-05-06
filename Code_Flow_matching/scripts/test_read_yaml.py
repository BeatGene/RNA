from utils import read_yaml

if __name__ == "__main__":
    config = read_yaml("config/RNA_test.yaml")
    print(config)