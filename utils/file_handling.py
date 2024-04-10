import os
import pickle

MODEL_DIR = "./models"


def load_pickle(experiment, run):
    file_path = os.path.join(MODEL_DIR, experiment, f"{run}.pkl")

    with open(file_path, "rb") as file:
        model = pickle.load(file)

    print(f"Loaded model from file '{file_path}'.")
    return model


def save_pickle(model, experiment, run):
    file_path = os.path.join(MODEL_DIR, experiment, f"{run}.pkl")

    try:
        os.mkdir(MODEL_DIR)
    except:
        pass

    try:
        os.mkdir(os.path.join(MODEL_DIR, experiment))
    except:
        pass

    with open(file_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Saved model to file '{file_path}'.")
