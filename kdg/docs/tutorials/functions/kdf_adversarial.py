import numpy as np
import matplotlib.pyplot as plt
import random
from kdg import kdf
import pandas as pd
import openml

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from art.utils import to_categorical


def get_adversarial_examples(model, x_attack, n_attack=20, nb_classes=2, idx=None):
    """
    Get adversarial examples from a trained model.
    """

    # Create a BB classifier: prediction function, num features, num classes
    def _predict(x):
        """Wrapper function to query black box"""
        return to_categorical(model.predict(x), nb_classes=nb_classes)

    art_classifier = BlackBoxClassifier(_predict, x_attack[0].shape, 2)

    # Create an attack model
    attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,
        max_iter=20,
        max_eval=1000,
        init_eval=10,
    )

    # Attack a random subset
    if idx is None:
        idx = random.sample(list(np.arange(0, len(x_attack))), n_attack)

    x_train_adv = attack.generate(x_attack[idx])
    return x_train_adv, idx, model


def plot_attack(model, x_train, y_train, x_train_adv, num_classes):
    """
    Utility function for visualizing how the attack was performed.
    """
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))

    colors = ["orange", "blue", "green"]

    for i_class in range(num_classes):

        # Plot difference vectors
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1], c="black", zorder=1)

        # Plot benign samples
        for i_class_2 in range(num_classes):
            axs[i_class].scatter(
                x_train[y_train == i_class_2][:, 0],
                x_train[y_train == i_class_2][:, 1],
                s=20,
                zorder=2,
                c=colors[i_class_2],
            )
        axs[i_class].set_aspect("equal", adjustable="box")

        # Show predicted probability as contour plot
        h = 0.05
        x_min, x_max = -2, 2
        y_min, y_max = -2, 2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i_class].reshape(xx.shape)
        im = axs[i_class].contourf(
            xx,
            yy,
            Z_proba,
            levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            vmin=0,
            vmax=1,
        )
        if i_class == num_classes - 1:
            cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
            plt.colorbar(im, ax=axs[i_class], cax=cax)

        # Plot adversarial samples
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c="red", marker="X")
        axs[i_class].set_xlim((x_min, x_max))
        axs[i_class].set_ylim((y_min, y_max))

        axs[i_class].set_title("class " + str(i_class))
        axs[i_class].set_xlabel("feature 1")
        axs[i_class].set_ylabel("feature 2")


def experiment(dataset_id, n_estimators=500, reps=5, n_attack=50):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    # Error checking: make sure no categorial or missing data
    if np.mean(is_categorical) > 0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return

    unique_classes, counts = np.unique(y, return_counts=True)
    test_sample = min(counts) // 3
    indx = []
    for label in unique_classes:
        indx.append(np.where(y == label)[0])
    max_sample = min(counts) - test_sample
    train_samples = np.logspace(
        np.log10(2), np.log10(max_sample), num=10, endpoint=True, dtype=int
    )

    train_samples = [train_samples[-1]]

    # Store statistics
    l2_kdf_list = []
    l2_rf_list = []
    linf_kdf_list = []
    linf_rf_list = []
    err_adv_rf_list = []
    err_adv_kdf_list = []
    err_rf = []
    err_kdf = []
    mc_rep = []
    samples_attack = []
    samples = []

    for train_sample in train_samples:
        for rep in range(reps):
            indx_to_take_train = []
            indx_to_take_test = []

            for ii, _ in enumerate(unique_classes):
                np.random.shuffle(indx[ii])
                indx_to_take_train.extend(list(indx[ii][:train_sample]))
                indx_to_take_test.extend(list(indx[ii][-test_sample : counts[ii]]))

            # Fit the estimators
            model_kdf = kdf(
                kwargs={
                    "n_estimators": n_estimators,
                    "min_samples_leaf": int(
                        np.ceil(X.shape[1] * 10 / np.log(train_sample))
                    ),
                }
            )
            model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
            proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_test])
            predicted_label_kdf = np.argmax(proba_kdf, axis=1)
            predicted_label_rf = np.argmax(proba_rf, axis=1)

            # Initial classification error
            err_rf.append(1 - np.mean(predicted_label_rf == y[indx_to_take_test]))
            err_kdf.append(1 - np.mean(predicted_label_kdf == y[indx_to_take_test]))

            # Begin adversarial attack code
            def _predict_kdf(x):
                """Wrapper to query black box"""
                proba_kdf = model_kdf.predict_proba(x)
                predicted_label_kdf = np.argmax(proba_kdf, axis=1)
                return to_categorical(
                    predicted_label_kdf,
                    nb_classes=len(np.unique(y[indx_to_take_train])),
                )

            def _predict_rf(x):
                """Wrapper to query blackbox for rf"""
                proba_rf = model_kdf.rf_model.predict_proba(x)
                predicted_label_rf = np.argmax(proba_rf, axis=1)
                return to_categorical(
                    predicted_label_rf, nb_classes=len(np.unique(y[indx_to_take_train]))
                )

            art_classifier_kdf = BlackBoxClassifier(
                _predict_kdf,
                X[indx_to_take_train][0].shape,
                len(np.unique(y[indx_to_take_train])),
            )
            art_classifier_rf = BlackBoxClassifier(
                _predict_rf,
                X[indx_to_take_train][0].shape,
                len(np.unique(y[indx_to_take_train])),
            )
            attack_rf = HopSkipJump(
                classifier=art_classifier_rf,
                targeted=False,
                max_iter=10,
                max_eval=1000,
                init_eval=10,
            )
            attack_kdf = HopSkipJump(
                classifier=art_classifier_kdf,
                targeted=False,
                max_iter=10,
                max_eval=1000,
                init_eval=10,
            )

            ### For computational reasons, attack a random subset that is identified correctly
            # Get indices of correctly classified samples common to both
            selection_idx = indx_to_take_train
            proba_kdf = model_kdf.predict_proba(X[selection_idx])
            proba_rf = model_kdf.rf_model.predict_proba(X[selection_idx])
            predicted_label_kdf = np.argmax(proba_kdf, axis=1)
            predicted_label_rf = np.argmax(proba_rf, axis=1)

            idx_kdf = np.where(predicted_label_kdf == y[selection_idx])[0]
            idx_rf = np.where(predicted_label_rf == y[selection_idx])[0]
            idx_common = list(np.intersect1d(idx_kdf, idx_rf))

            # Randomly sample from the common indices
            if n_attack > len(idx_common):
                n_attack = len(idx_common)
            idx = random.sample(idx_common, n_attack)
            if n_attack == 0:
                return

            # Generate samples
            x_adv_kdf = attack_kdf.generate(X[selection_idx][idx])
            x_adv_rf = attack_rf.generate(X[selection_idx][idx])

            # Compute norms
            l2_kdf = np.mean(
                np.linalg.norm(X[selection_idx][idx] - x_adv_kdf, ord=2, axis=1)
            )
            l2_rf = np.mean(
                np.linalg.norm(X[selection_idx][idx] - x_adv_rf, ord=2, axis=1)
            )
            linf_rf = np.mean(
                np.linalg.norm(X[selection_idx][idx] - x_adv_rf, ord=np.inf, axis=1)
            )
            linf_kdf = np.mean(
                np.linalg.norm(X[selection_idx][idx] - x_adv_kdf, ord=np.inf, axis=1)
            )

            ### Classification
            # Make adversarial prediction
            proba_rf = model_kdf.rf_model.predict_proba(x_adv_rf)
            predicted_label_rf_adv = np.argmax(proba_rf, axis=1)
            err_adv_rf = 1 - np.mean(predicted_label_rf_adv == y[selection_idx][idx])

            proba_kdf = model_kdf.predict_proba(x_adv_kdf)
            predicted_label_kdf_adv = np.argmax(proba_kdf, axis=1)
            err_adv_kdf = 1 - np.mean(predicted_label_kdf_adv == y[selection_idx][idx])

            l2_kdf_list.append(l2_kdf)
            l2_rf_list.append(l2_rf)
            linf_kdf_list.append(linf_kdf)
            linf_rf_list.append(linf_rf)
            err_adv_kdf_list.append(err_adv_kdf)
            err_adv_rf_list.append(err_adv_rf)

            mc_rep.append(rep)
            samples_attack.append(n_attack)
            samples.append(train_sample)

    df = pd.DataFrame()
    df["l2_kdf"] = l2_kdf_list
    df["l2_rf"] = l2_rf_list
    df["linf_kdf"] = linf_kdf_list
    df["linf_rf"] = linf_rf_list
    df["err_kdf"] = err_kdf
    df["err_rf"] = err_rf
    df["err_adv_kdf"] = err_adv_kdf_list
    df["err_adv_rf"] = err_adv_rf_list
    df["rep"] = mc_rep
    df["samples_attack"] = samples_attack
    df["samples"] = samples

    return (dataset_id, df)


def combine_results(data):
    """Given the results from the experiment, display
    the average metrics across all datasets."""
    fname = []
    l2_rf = []
    l2_kdf = []
    linf_rf = []
    linf_kdf = []
    err_adv_kdf = []
    err_adv_rf = []
    delta_adv_err_list = []
    delta_adv_l2_list = []
    delta_adv_linf_list = []
    for el in data:
        dataset_id = el[0]
        df = el[1]
        l2_mean_rf = df["l2_rf"].mean()
        linf_mean_rf = df["linf_rf"].mean()

        l2_mean_kdf = df["l2_kdf"].mean()
        linf_mean_kdf = df["linf_kdf"].mean()

        err_adv_mean_kdf = df["err_adv_kdf"].mean()
        err_adv_mean_rf = df["err_adv_rf"].mean()

        err_mean_kdf = df["err_kdf"].mean()
        err_mean_rf = df["err_rf"].mean()

        delta_adv_err = np.mean(df["err_adv_kdf"] - df["err_adv_rf"])
        delta_adv_l2 = np.mean(df["l2_kdf"] - df["l2_rf"])
        delta_adv_linf = np.mean(df["linf_kdf"] - df["linf_rf"])

        fname.append(dataset_id)
        l2_rf.append(l2_mean_rf)
        l2_kdf.append(l2_mean_kdf)
        linf_rf.append(linf_mean_rf)
        linf_kdf.append(linf_mean_kdf)
        err_adv_kdf.append(err_adv_mean_kdf)
        err_adv_rf.append(err_adv_mean_rf)
        delta_adv_err_list.append(delta_adv_err)
        delta_adv_l2_list.append(delta_adv_l2)
        delta_adv_linf_list.append(delta_adv_linf)

    df = pd.DataFrame()
    df["ID"] = fname
    df["l2_kdf"] = l2_kdf
    df["l2_rf"] = l2_rf
    df["linf_kdf"] = linf_kdf
    df["linf_rf"] = linf_rf
    df["err_adv_kdf"] = err_adv_kdf
    df["err_adv_rf"] = err_adv_rf
    df["delta_adv_err"] = delta_adv_err_list
    df["delta_adv_l2"] = delta_adv_l2_list
    df["delta_adv_linf"] = delta_adv_linf_list
    return df
