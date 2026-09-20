import matplotlib.pyplot as plt

from pipeline import load_dataset
from model import run_one_trial
from evaluation import evaluate_models


# ==========================
# Settings
# ==========================

TRAIN_ITEM_NUMS = range(4, 9)

TRIAL = 0   # 確認用のseed


# ==========================
# Load data
# ==========================

pram, spam_pairs = load_dataset()


# ==========================
# Run & Plot
# ==========================

for train_num in TRAIN_ITEM_NUMS:

    print("=" * 50)
    print(f"Train stimuli = {train_num}")
    print("=" * 50)


    result = run_one_trial(
        spam_pairs=spam_pairs,
        pram=pram,
        train_item_num=train_num,
        random_state=TRIAL,
    )


    metrics = evaluate_models(result)


    for model_name in [
        "Linear",
        "Probit",
        "Logit",
    ]:

        data = metrics[model_name]["pair_result"]

        y_true = data["True"]
        y_pred = data["Predicted"]


        plt.figure(
            figsize=(5,5)
        )


        plt.scatter(
            y_true,
            y_pred,
        )


        lim_min = min(
            y_true.min(),
            y_pred.min()
        )

        lim_max = max(
            y_true.max(),
            y_pred.max()
        )


        plt.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            linestyle="--",
        )


        plt.xlabel(
            "Observed PRaM"
        )

        plt.ylabel(
            "Predicted PRaM"
        )


        plt.title(
            f"{model_name} | Train stimuli={train_num}"
        )


        plt.xlim(
            lim_min,
            lim_max
        )

        plt.ylim(
            lim_min,
            lim_max
        )


        plt.grid()

        plt.show()
