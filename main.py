# Press the green button in the gutter to run the script.
import json

import pandas as pd
from colorama import Fore, init

from Modules.AnomalyDetection import clear_anomalies_directory, find_anomalies, find_anomalies_univariate, \
    get_sensor_data, train_models, \
    train_univariate
from Modules.Database import save_df_to_database, save_new_data_to_db
from Modules.scheduler import my_schedule

init(autoreset=True)
sensors = ['addvantage', 'Teros_12', 'Triscan', 'Scan_chlori', 'Aquatroll', 'Proteus_infinite', 'ATMOS']


def find_ano():
    for sensor in sensors:
        train_models(sensor)
        df = find_anomalies(sensor)
        print(df.shape)
        #
        save_df_to_database(df=df, table_name=sensor + "_Anomalies")


def find_ano2(clear=True):
    if clear:
        clear_anomalies_directory()

    for sensor in sensors:
        print(f'{Fore.GREEN}sensor: {sensor}')
        data = get_sensor_data(sensor)
        data = data.dropna()
        data.drop(columns=['application_group'], inplace=True)

        for column in data.columns:
            train = data[column][:-24]
            test = data[column][-24:]
            out = pd.DataFrame()
            a = train_univariate(sensor, column, train.values.reshape(-1, 1))

            for clf, clf_name in a:
                print(f'{Fore.RED}clf_name: {clf_name}')

                outliers = find_anomalies_univariate(clf, clf_name, sensor, column, test)

                if outliers is not None:
                    out = pd.concat([out, outliers.reset_index()], axis=0, ignore_index=True)

            if out.shape[0] > 0:
                out = out.set_index('timestamp')
                out = out.drop_duplicates(keep='first')
                out = out.sort_index()

                out = out.reset_index()
                to_save = out.copy()
                out.reset_index(inplace=True)
                out.drop(columns=['index'], inplace=True)
                res = json.loads(out.to_json(orient='records', date_format='iso'))

                out.to_csv(f'Anomalies/{sensor}_{column}.csv', index=False)

                if out.shape[0] > 0:
                    save_new_data_to_db(df=to_save, table_name='anomalies')


if __name__ == '__main__':
    # find_ano2()
    my_schedule(find_ano2)
