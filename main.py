# Press the green button in the gutter to run the script.
from Modules.AnomalyDetection import find_anomalies, train_models
from Modules.Database import save_df_to_database
from Modules.scheduler import my_schedule

sensors = ['Teros_12', 'Triscan', 'Scan_chlori', 'Aquatroll', 'Proteus_infinite', 'ATMOS', 'addvantage']


def find_ano():
    for sensor in sensors:
        train_models(sensor)
        df = find_anomalies(sensor)
        print(df.shape)
        #
        save_df_to_database(df=df, table_name=sensor + "_Anomalies")


if __name__ == '__main__':
    # find_ano()
    my_schedule(find_ano)
