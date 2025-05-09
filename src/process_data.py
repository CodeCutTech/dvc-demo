import warnings
from datetime import date

import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
import hydra
from pathlib import Path

warnings.simplefilter(action="ignore", category=UserWarning)


def read_data(config: DictConfig) -> pd.DataFrame:
    return pd.read_csv(config.raw_data.path)


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def get_age(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(age=df["Year_Birth"].apply(lambda row: date.today().year - row))


def get_total_children(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(total_children=df["Kidhome"] + df["Teenhome"])


def get_total_purchases(df: pd.DataFrame) -> pd.DataFrame:
    purchases_columns = df.filter(like="Purchases", axis=1).columns
    return df.assign(total_purchases=df[purchases_columns].sum(axis=1))


def get_enrollment_years(df: pd.DataFrame) -> pd.DataFrame:
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    return df.assign(enrollment_years=2022 - df["Dt_Customer"].dt.year)


def get_family_size(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    return df.assign(
        family_size=df["Marital_Status"].map(config.process.family_size)
        + df["total_children"]
    )


def drop_features(df: pd.DataFrame, config: DictConfig):
    df = df[config.process.keep_columns]
    return df


def drop_outliers(df: pd.DataFrame, config: DictConfig):
    column_threshold = dict(config.process.remove_outliers_threshold)
    for col, threshold in column_threshold.items():
        df = df[df[col] < threshold]
    return df.reset_index(drop=True)


def get_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df)

    return scaler


def scale_features(df: pd.DataFrame, scaler: StandardScaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


def save_process_data(df: pd.DataFrame, config: DictConfig):
    Path(config.intermediate.path).parent.mkdir(exist_ok=True)
    df.to_csv(config.intermediate.path, index=False)


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    df = read_data(config)
    df = (
        df.pipe(drop_na)
        .pipe(get_age)
        .pipe(get_total_children)
        .pipe(get_total_purchases)
        .pipe(get_enrollment_years)
        .pipe(get_family_size, config)
        .pipe(drop_features, config)
        .pipe(drop_outliers, config)
    )
    scaler = get_scaler(df)
    df = scale_features(df, scaler)
    save_process_data(df, config)


if __name__ == "__main__":
    process_data()
