#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Analysis for SPY Data
==========================

Loads the SPYV3.csv dataset, drops non-predictive columns, encodes
categorical variables, scales numeric features, and performs Principal
Component Analysis with automatic component selection. Produces a
scree plot, a loadings heatmap, a biplot, and CSV exports of the
variance, loadings and transformed scores.

Author: Pablo Beret
Created on Sat Sep 15 14:03:18 2018 (Updated version)
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pca_analysis")

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

DROP_COLUMNS = [
    "FECHA", "OPEN", "MAX", "MIN", "CLOSE", "CLASIFICADOR",
    "FECHA.year", "FECHA.day-of-month", "FECHA.day-of-week",
]
CATEGORICAL_COLUMNS = ["39", "41", "43", "168", "172"]
DEFAULT_VARIANCE_THRESHOLD = 0.90

@dataclass
class PCAConfig:
    data_path: str = "SPYV3.csv"
    scaler: str = "standard"  # "standard" o "minmax"
    n_components: Optional[int] = None
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    loading_threshold: float = 0.30
    output_dir: Path = Path("output")

# ----------------------------------------------------------------------
# Carga y preprocesado
# ----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    df = pd.read_csv(file_path, sep=",")
    logger.info("Dataset cargado desde %s: %s filas x %s columnas", file_path, *df.shape)

    existing_drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    logger.info("Columnas eliminadas: %s", existing_drop_cols)

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(
            "El dataset contiene %d valores nulos; se eliminarán las filas afectadas "
            "para evitar errores en el escalado y en el PCA.",
            n_missing,
        )
        df = df.dropna().reset_index(drop=True)
        logger.info("Filas restantes tras eliminar nulos: %d", len(df))

    return df

def build_preprocessor(df: pd.DataFrame, scaler_name: str) -> ColumnTransformer:
    """
    Codificación y escalado correctos para PCA:

    - One-hot encoding en vez de pd.factorize para las variables
      categóricas: factorize asigna códigos enteros arbitrarios que
      implican una relación de orden inexistente entre categorías
      nominales, distorsionando las distancias en las que se basa PCA.
    - Escalado de las variables numéricas: PCA se basa en la matriz de
      covarianzas, así que sin escalar (o con un escalado que no centra
      en media 0, como MinMax) las variables de mayor varianza dominan
      artificialmente los componentes.
    """
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    if scaler_name == "minmax":
        logger.warning(
            "Usando MinMaxScaler: no centra los datos en media 0. StandardScaler "
            "es en general la opción más adecuada para PCA."
        )
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor

# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------

def select_n_components(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(explained_variance_ratio)
    n = int(np.searchsorted(cumulative, threshold) + 1)
    return min(n, len(explained_variance_ratio))


def run_pca(df: pd.DataFrame, config: PCAConfig) -> tuple[PCA, np.ndarray, list[str]]:
    preprocessor = build_preprocessor(df, config.scaler)
    x_transformed = preprocessor.fit_transform(df)
    feature_names = list(preprocessor.get_feature_names_out())

    max_components = min(x_transformed.shape)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(x_transformed)

    if config.n_components is not None:
        n_components = min(config.n_components, max_components)
        logger.info("Número de componentes fijado por el usuario: %d", n_components)
    else:
        n_components = select_n_components(pca_full.explained_variance_ratio_, config.variance_threshold)
        logger.info(
            "Componentes necesarios para explicar >= %.0f%% de la varianza: %d",
            config.variance_threshold * 100,
            n_components,
        )

    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_transformed)
    return pca, x_pca, feature_names

def report_variance(pca: PCA, output_dir: Path) -> pd.DataFrame:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    variance_df = pd.DataFrame(
        {
            "component": [f"PC-{i + 1}" for i in range(len(ratios))],
            "explained_variance_ratio": ratios,
            "cumulative_variance": cumulative,
        }
    )
    logger.info("Varianza explicada:\n%s", variance_df.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    variance_df.to_csv(output_dir / "explained_variance.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(variance_df["component"], variance_df["explained_variance_ratio"], color="#2E86AB", label="Individual")
    ax1.set_ylabel("Varianza explicada")
    ax1.set_xlabel("Componente principal")
    ax2 = ax1.twinx()
    ax2.plot(
        variance_df["component"], variance_df["cumulative_variance"],
        color="#E63946", marker="o", label="Acumulada",
    )
    ax2.set_ylabel("Varianza acumulada")
    ax2.set_ylim(0, 1.05)
    fig.suptitle("Scree plot")
    fig.tight_layout()
    fig.savefig(output_dir / "scree_plot.png", dpi=150)
    plt.close(fig)

    return variance_df

def compute_loadings(pca: PCA, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    n_components = pca.n_components_
    loadings = pd.DataFrame(
        np.transpose(pca.components_),
        columns=[f"PC-{i + 1}" for i in range(n_components)],
        index=feature_names,
    )
    logger.info("PCA loadings:\n%s", loadings.to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(output_dir / "pca_loadings.csv")

    fig, ax = plt.subplots(figsize=(max(8, n_components * 1.5), max(6, len(feature_names) * 0.3)))
    im = ax.imshow(loadings.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    fig.colorbar(im, ax=ax, label="Loading")
    ax.set_title("PCA loadings heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_loadings_heatmap.png", dpi=150)
    plt.close(fig)

    return loadings

def filter_high_loadings(loadings: pd.DataFrame, threshold: float, output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Filtra por valor absoluto: una variable con loading -0.35 contribuye
    tanto al componente como una con +0.35. El script original solo
    filtraba loadings positivos por encima del umbral, descartando
    silenciosamente contribuciones negativas relevantes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for pc in loadings.columns:
        filtered = loadings.loc[loadings[pc].abs() >= threshold, [pc]].sort_values(pc, ascending=False)
        results[pc] = filtered
        logger.info("Variables con |loading| >= %.2f en %s:\n%s", threshold, pc, filtered.to_string())
        filtered.to_csv(output_dir / f"high_loadings_{pc}.csv")
    return results

def plot_biplot(x_pca: np.ndarray, loadings: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Biplot de PC-1 vs PC-2 con las variables de mayor contribución conjunta."""
    if x_pca.shape[1] < 2:
        logger.info("Menos de 2 componentes disponibles; se omite el biplot.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.3, s=10, color="#2E86AB")

    magnitude = np.sqrt(loadings["PC-1"] ** 2 + loadings["PC-2"] ** 2)
    top_features = magnitude.sort_values(ascending=False).head(top_n).index
    scale = np.abs(x_pca[:, :2]).max() * 0.9

    for feature in top_features:
        x_load, y_load = loadings.loc[feature, "PC-1"], loadings.loc[feature, "PC-2"]
        ax.arrow(0, 0, x_load * scale, y_load * scale, color="#E63946", alpha=0.8, head_width=scale * 0.02)
        ax.text(x_load * scale * 1.1, y_load * scale * 1.1, str(feature), color="#E63946", fontsize=8)

    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    ax.set_title("PCA biplot (PC-1 vs PC-2)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_biplot.png", dpi=150)
    plt.close(fig)

def save_scores(x_pca: np.ndarray, output_dir: Path) -> None:
    scores_df = pd.DataFrame(x_pca, columns=[f"PC-{i + 1}" for i in range(x_pca.shape[1])])
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_dir / "pca_scores.csv", index=False)
    logger.info("Scores de PCA guardados en %s", output_dir / "pca_scores.csv")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> PCAConfig:
    parser = argparse.ArgumentParser(description="Análisis de Componentes Principales (PCA) sobre datos SPY.")
    parser.add_argument("--data-path", type=str, default="SPYV3.csv")
    parser.add_argument("--scaler", type=str, choices=["standard", "minmax"], default="standard")
    parser.add_argument("--n-components", type=int, default=None, help="Fija el número de componentes manualmente.")
    parser.add_argument(
        "--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
        help="Varianza acumulada mínima a explicar si no se fija --n-components.",
    )
    parser.add_argument(
        "--loading-threshold", type=float, default=0.30,
        help="Umbral absoluto para marcar variables de alta contribución.",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    return PCAConfig(
        data_path=args.data_path,
        scaler=args.scaler,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
        loading_threshold=args.loading_threshold,
        output_dir=Path(args.output_dir),
    )

def main() -> None:
    config = parse_args()
    df = load_data(config.data_path)
    pca, x_pca, feature_names = run_pca(df, config)

    report_variance(pca, config.output_dir)
    loadings = compute_loadings(pca, feature_names, config.output_dir)
    filter_high_loadings(loadings, config.loading_threshold, config.output_dir)
    plot_biplot(x_pca, loadings, config.output_dir)
    save_scores(x_pca, config.output_dir)

if __name__ == "__main__":
    main()
