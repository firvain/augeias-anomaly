import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

from Modules.Database import get_data_from_augeias_postgresql

sns.set(rc={'figure.figsize': (11.69, 8.27)})

outliers_fraction = 0.05
random_state = np.random.RandomState(42)
classifiers = {
    'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    # 'Empirical Cumulative Distribution Functions': ECOD(contamination=outliers_fraction),
    'Auto Encoder with Outlier Detection': AutoEncoder(contamination=0.1, random_state=random_state,
                                                       hidden_neurons=[4, 2, 2, 4], verbose=0)
}


def train_models(table_name: str):
    sql = f"""select * from "{table_name}" order by timestamp"""
    data = get_data_from_augeias_postgresql(table_name, sql)
    data.dropna(inplace=True)
    test = data.iloc[:-24]

    # train, test = train_test_split(data, test_size=0.2)

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # print(i, clf)
        clf.fit(test)
        dump(clf, 'models/' + table_name + '_' + clf_name + '.joblib')


def find_anomalies(table_name: str):
    sql = f"""select * from "{table_name}" order by timestamp DESC LIMIT 24"""
    data = get_data_from_augeias_postgresql(table_name, sql)

    # train, test = train_test_split(data, test_size=0.2)
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    print(data)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    num_mes = data.shape[0]
    df = pd.DataFrame()
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # print(i)

        clf = load('models/' + table_name + '_' + clf_name + '.joblib')

        # get the prediction on the test data
        y_test_pred = clf.predict(data.values)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(data.values)  # outlier scores

        outliers = data.iloc[y_test_pred == 1]

        out = outliers.sort_index()

        df = pd.concat([df, out], axis=0)

        num_outliers = outliers.shape[0]
        # dfm = outliers.reset_index().melt('timestamp', var_name="measurements", value_name="val")
        #
        # g = sns.catplot(x="timestamp", y="val", hue='measurements', data=dfm, legend_out=True)
        # ax = g.axes
        # textstr = '\n'.join((
        #     f'Num measurements : {num_mes}',
        #     f'Num anomalies: {num_outliers}'))
        # ax = plt.gca()
        # # get current xtick labels
        # xticks = ax.get_xticks()
        # ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
        # # convert all xtick labels to selected format from ms timestamp
        # ax.set_xticklabels([dfm.iloc[tm]['timestamp'].strftime('%d-%m-%y\n %H:%M') for tm in xticks],
        #                    rotation=90, fontsize=6)
        # plt.text(.78, .9, textstr, fontsize=8, transform=plt.gcf().transFigure)
        # ax.grid(True, axis='both')
        # g.fig.suptitle(clf_name)
        # plt.show()

    print(df.columns)
    print(df.shape)
    df = df.groupby(df.index).first()
    return df
