import seaborn as sn
import pandas as pd
import numpy as np
from numpy import newaxis
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    average_precision_score,
    recall_score,
)
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def plot_heat_map(ax, X, Y, Z, xlabel, ylabel, format="d", title="Heat Map"):

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(
        Z,
        annot=True,
        fmt=format,
        annot_kws={"size": 12},
        cmap="YlGnBu",
        xticklabels=X,
        yticklabels=Y,
        ax=ax,
    )  # font size
    ax.set_title(title)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)

def plot_classification_report(cr, ax, title):
    df = pd.DataFrame(cr).iloc[:, :].T
    df = df.round(2)
    mask = np.zeros(df.shape)
    mask[:, -1] = True 
    sn.heatmap(df, 
            mask=mask,
            annot=True, 
            cmap="YlGnBu",
            fmt='g',
            ax=ax)       
    for (j,i), label in np.ndenumerate(df.values):
        if i == 3:
            ax.text(i+0.5, j+0.5, label,
                fontdict=dict(ha='center',  va='center',
                                 color='g', fontsize=18))

    ax.set_title(title)

def plot_confusion_matrix(
    ax, cm, labels, normalize=False, title="Confusion Matrix"
):
    format = "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, newaxis]
        format = ".2f"

    plot_heat_map(
        ax, labels, labels, cm, "Predicted", "True Classes", format, title
    )


# evalute classifer
def eval_conf(
    classifier,
    x_train,
    y_train,
    x_test,
    y_test,
    plot_conf=False,
    labels=[],
    title="",
    ax=None,
    normalized_conf=True,
    figsize=(10, 8),
):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return report(y_test, y_pred)


def report(
    y_test,
    y_pred,
    plot_conf=True,
    labels=[],
    ax=None,
    normalized_conf=True,
    figsize=(10, 18),
    title="",
    image=""
):
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True, labels=labels, zero_division=1)
    # plot confusion matrix
    if not labels:
        labels = list(set(y_test))
    if plot_conf:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        precision, recall, fscore, support = score(
            y_test, y_pred, average="macro"
        )
        if ax == None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
        if title == "":
            title += f"acc-{acc.round(2)} pr-{precision.round(2)} re-{recall.round(2)} f1-{fscore.round(2)}"
        plot_classification_report(cr, ax1, title)
        plot_confusion_matrix(ax2, cm, labels, normalized_conf, title)
        if image:
            plt.savefig(image)

    return acc, cr


def evalsplit_conf(
    classifier,
    x_train,
    y_train,
    test_size,
    plot_conf=False,
    labels=[],
    title="",
    ax=None,
    normalized_conf=True,
    figsize=(10, 8),
):
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=test_size, random_state=42
    )
    acc, cr = eval_conf(
        classifier,
        x_train,
        y_train,
        x_test,
        y_test,
        plot_conf,
        labels,
        title,
        ax,
        normalized_conf,
        figsize,
    )
    return acc, cr


def plot_dict(dict, ax=None):
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 8))
    lists = sorted(dict.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    ax.plot(x, y)
