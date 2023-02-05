# Press the green button in the gutter to run the script.
import json
from pathlib import Path

import pandas as pd
from colorama import Fore, init
from matplotlib import pyplot as plt

from Modules.AnomalyDetection import classifiers, clear_anomalies_directory, clear_models_directory, \
    find_anomalies_univariate, \
    get_last_24h_data, get_sensor_data, train_univariate

init(autoreset=True)
sensors = ['Teros_12', 'Triscan', 'Scan_chlori', 'Aquatroll', 'Proteus_infinite', 'ATMOS', 'addvantage', ]


# def find_ano():
#     for sensor in sensors:
#         train_models(sensor)
#         df = find_anomalies(sensor)
#         print(df.shape)
#         #
#         save_df_to_database(df=df, table_name=sensor + "_Anomalies")


def find_ano2(clear=True, should_train=True):
    if clear:
        clear_anomalies_directory()
        clear_models_directory()

    for sensor in sensors:
        print(f'{Fore.GREEN}*' * 100)
        print(f'{Fore.GREEN}sensor: {sensor}')

        data = get_sensor_data(sensor)
        data = data.dropna()
        data.drop(columns=['application_group'], inplace=True)
        sensor_dataframe = pd.DataFrame()

        for column in data.columns:
            print(f'{Fore.BLUE}column: {column}')

            train = data[column][:-24]
            test = data[column][-24:]

            out = pd.DataFrame()
            if should_train:
                a = train_univariate(sensor, column, train.values.reshape(-1, 1))
            else:
                a = classifiers.items()

            for clf, clf_name in a:
                # print(f'{Fore.YELLOW}clf_name: {clf_name}')
                # print(sensor, column)
                test = get_last_24h_data(sensor)

                test = test[column]
                test = test.dropna()

                if test.shape[0] > 0:
                    outliers = find_anomalies_univariate(sensor, column, test)

                    if outliers is not None:
                        out = pd.concat([out, outliers.reset_index()], axis=0, ignore_index=True)

            if out.shape[0] > 0:
                out = out.set_index('timestamp')
                out = out.drop_duplicates(keep='first')
                out = out.sort_index()

                out = out.reset_index()
                sensor_dataframe = pd.concat([sensor_dataframe, pd.DataFrame(out)], axis=0, ignore_index=True)

                to_save = out.copy()
                out.reset_index(inplace=True)
                out.drop(columns=['index'], inplace=True)
                res = json.loads(out.to_json(orient='records', date_format='iso'))

                # out.to_csv(f'Anomalies/{sensor}_{column}.csv', index=False)
                #
                # if out.shape[0] > 0:
                #     save_new_data_to_db(df=to_save, table_name='anomalies')
        print(sensor_dataframe.shape)
        if sensor_dataframe.shape[0] > 0:
            sensor_dataframe.drop(columns=['sensor'], inplace=True)
            print(sensor_dataframe)
            sensor_dataframe = sensor_dataframe.pivot_table(index=['timestamp'], columns='variable',
                                                            values='value', aggfunc='first').reset_index()
            sensor_dataframe = sensor_dataframe.set_index('timestamp')
            sensor_dataframe = sensor_dataframe.sort_index()
            sensor_dataframe.to_csv(f'Anomalies/{sensor}.csv')

            sensor_dataframe.plot(subplots=True, figsize=(20, 20), title=sensor,
                                  sharex=True, ls="none", marker="o",
                                  y=[_ for _ in sensor_dataframe.columns], grid=True)
            plt.tight_layout()

            Path(f'Anomalies/PNG/').mkdir(parents=True, exist_ok=True)
            plt.savefig(f'Anomalies/PNG/{sensor}.png')
            plt.show()
            try:
                pass
                # save_df_to_database(df=sensor_dataframe, table_name=sensor + "_Anomalies")
            except Exception as e:
                print(e)
                pass


if __name__ == '__main__':
    find_ano2(should_train=False, clear=False)
    # my_schedule(find_ano2)
