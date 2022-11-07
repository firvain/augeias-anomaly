import seaborn as sns
from matplotlib import pyplot as plt
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.suod import SUOD
from sklearn.model_selection import train_test_split

from Modules.Database import get_data_from_augeias_postgresql

sns.set(rc={'figure.figsize': (11.7, 8.27)})


def find_anomalies(table_name: str):
    data = get_data_from_augeias_postgresql(table_name)

    train, test = train_test_split(data, test_size=0.2)
    print(data.shape)
    print(train.shape)
    print(test.shape)
    # train ECOD detector
    clf_name = 'KNN'
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35),
                     COPOD(), IForest(n_estimators=100),
                     IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
               verbose=False)
    # clf = KNN()

    # you could try parallel version as well.
    # clf = ECOD(n_jobs=2)
    clf.fit(data)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(data)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(data)  # outlier scores

    outliers = data.iloc[y_test_pred == 1]
    print(outliers)
    dfm = outliers.reset_index().melt('timestamp', var_name="measurements", value_name="val")
    print(dfm)
    g = sns.catplot(x="timestamp", y="val", hue='measurements', data=dfm)

    plt.grid()
    plt.show()
    # # evaluate and print the results
    # print("\nOn Training Data:")
    # evaluate_print(clf_name, train, y_train_scores)
    # # print("\nOn Test Data:")
    # evaluate_print(clf_name, test, y_test_scores)
    # #
    # # visualize the results
    # visualize(clf_name, train, test, y_train_pred,
    #           y_test_pred, show_figure=True, save_figure=False)
