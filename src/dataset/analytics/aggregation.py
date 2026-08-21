from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...schema import DatasetConfig
from ..processors.factory import DatasetLoaderFactory


class DataConcatenator:
    def __init__(self, configs: list[DatasetConfig]):
        self.configs = configs
        self.roots = [Path(config.root) for config in configs]

    def get_all_files(self) -> list[Path]:
        all_files = []
        for root in self.roots:
            files = sorted(root.rglob("*.wav"))
            all_files.extend(files)
        return all_files

    def aggregate_features(
        self, all_data: list[list[np.ndarray]]
    ) -> list[dict[str, float]]:

        aggregated_features = []
        feature_names = self.configs[0].feature_names

        for features in all_data:
            row = {}

            if len(features) != len(feature_names):
                raise ValueError("Mismatch between features and feature_names")

            for name, feat in zip(feature_names, features, strict=True):
                # [F, T] → mel, mfcc, contrast
                if feat.ndim == 2:
                    for j in range(feat.shape[0]):
                        signal = feat[j]
                        row[f"{name}_{j}_mean"] = float(signal.mean())
                        row[f"{name}_{j}_std"] = float(signal.std())

                # [1, T] → rms, zcr
                elif feat.ndim == 2 and feat.shape[0] == 1:
                    signal = feat[0]
                    row[f"{name}_mean"] = float(signal.mean())
                    row[f"{name}_std"] = float(signal.std())

                # [T] → pitch
                elif feat.ndim == 1:
                    row[f"{name}_mean"] = float(feat.mean())
                    row[f"{name}_std"] = float(feat.std())

                else:
                    raise ValueError(f"Unsupported shape {feat.shape}")

            aggregated_features.append(row)

        return aggregated_features

    def get_agg_data(
        self,
    ) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
        all_data = []
        all_metadata = []
        for config in self.configs:
            loader = DatasetLoaderFactory.create(config)
            data, metadata = loader.load(feature_mode="multi_channel")
            data = self.aggregate_features(data)
            all_data.extend(data)
            all_metadata.extend(metadata)
        return all_data, all_metadata

    def to_dataframe(
        self, agg_data: list[dict[str, float]], metadata: list[dict[str, Any]]
    ):
        df_features = pd.DataFrame(agg_data)
        df_metadata = pd.DataFrame(metadata)
        return pd.concat([df_metadata, df_features], axis=1)
