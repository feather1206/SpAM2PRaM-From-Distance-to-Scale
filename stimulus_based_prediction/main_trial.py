# main_trial.py

from pipeline import load_dataset

from model import run_one_trial
from evaluation import evaluate_models, print_result


def main():

    print("Loading dataset...")

    pram, spam_pairs = load_dataset()

    print("Running one trial...")

    result = run_one_trial(
        spam_pairs=spam_pairs,
        pram=pram,
        train_item_num=6,
        random_state=42,
    )

    if result is None:
        print("Training failed.")
        return

    metrics = evaluate_models(result)

    print_result(metrics)


if __name__ == "__main__":
    main()
