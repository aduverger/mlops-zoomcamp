import pickle
import pandas as pd
import numpy as np
import sys

categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename: str) -> pd.DataFrame:
    print(f"Reading data from {filename}...")
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def predict(df: pd.DataFrame, year: int, month: int, model_path: str = 'model.bin') -> np.array:
    print(f"Predicting duration with model {model_path}...")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"Mean duration predicted is: {y_pred.mean()}")
    return y_pred


def save_data(df: pd.DataFrame, y_pred: np.array, output_file: str) -> None:
    print(f"Saving prediction to {output_file}...")
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)
    y_pred = predict(df, year, month)
    save_data(df, y_pred, output_file)


if __name__ == "__main__":
    run()
