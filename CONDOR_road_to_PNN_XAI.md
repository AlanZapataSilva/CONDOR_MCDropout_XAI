## Deep Learning for extensive air showers reconstruction: an interpretable multi-task CNN-Transformer model for the CONDOR Observatory

This notebook mirrors the pipeline described in the paper. Multi-task CNN-Transformer for (1) gamma/hadron classification, (2) zenith angle reconstruction, (3) energy estimation.


### Contributions and Reference

- Joint multi-task learning for classification + regression (angle, energy).
- Hybrid sequence + global branch with attention.
- Physics-driven feature engineering and balanced splits.

**Paper DOI:** _Coming soon_

**Author:** _Luis F. Navarro_ / 📧 luis.navarrof@usm.cl



```python
import os
import json
import pickle
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, callbacks, Input, Model
from typing import Any, Dict, Optional, Union, cast  # Importamos cast
import gdown
import logging
from datetime import datetime
import hashlib
import mlflow
from mlflow.models import infer_signature
import mlflow.keras

from scipy.stats import pearsonr, spearmanr, norm

from tqdm.notebook import tqdm

```

### Classes



```python
class MCSpatialDropout1D(tf.keras.layers.SpatialDropout1D):
    """
    SpatialDropout1D siempre activo para inferencia MC.

    Elimina canales enteros (eje de features) a través de TODOS los timesteps
    simultáneamente. Correcto para tensores (B, T, C) procedentes de Conv1D,
    donde timesteps adyacentes están correlacionados por el campo receptivo.

    Por qué no MCDropout estándar en tensores CNN:
        Con kernel_size=7, RF_efectivo≈19. Eliminar un elemento (b,t,c) no elimina la información de t porque posiciones t±1,...,t±6 la contienen también. SpatialDropout1D elimina el canal c completo — la Conv1D no puede recuperar un canal eliminado desde ningún vecino.
    """

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

    def get_config(self):
        return super().get_config()

```

### Functions



```python
# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN Y MÉTRICAS BAYESIANAS
# ─────────────────────────────────────────────────────────────────────────────


def temperature_scale(p_pred: np.ndarray, T: float) -> np.ndarray:
    """Aplica temperature scaling: σ(logit(p) / T)."""
    eps = 1e-7
    p_clip = np.clip(p_pred, eps, 1 - eps)
    logits = np.log(p_clip / (1 - p_clip))
    return 1.0 / (1.0 + np.exp(-logits / T))


def find_optimal_temperature(
    p_val: np.ndarray,
    y_val: np.ndarray,
    T_bounds: tuple[float, float] = (0.05, 20.0),
) -> tuple[float, float]:
    """Encuentra T* minimizando NLL en validación."""
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        p_cal = temperature_scale(p_val, T)
        return float(
            -np.mean(
                y_val * np.log(p_cal + 1e-8) + (1 - y_val) * np.log(1 - p_cal + 1e-8)
            )
        )

    result = minimize_scalar(
        nll, bounds=T_bounds, method="bounded", options={"xatol": 1e-5}
    )
    logging.info("Temperature Scaling — T*=%.4f  NLL_val=%.4f", result.x, result.fun)
    return float(result.x), float(result.fun)


def compute_ece(
    p_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error sobre bins uniformes en [0,1]."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(p_pred)) * abs(
            p_pred[mask].mean() - y_true[mask].mean()
        )
    return float(ece)


def compute_brier_score(
    p_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Brier Score y descomposición REL / RES / UNC."""
    bs = float(np.mean((p_pred - y_true) ** 2))
    y_bar = y_true.mean()
    rel, res = 0.0, 0.0
    for lo, hi in zip(*[np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        n_k = mask.sum()
        p_k = p_pred[mask].mean()
        y_k = y_true[mask].mean()
        rel += (n_k / len(y_true)) * (p_k - y_k) ** 2
        res += (n_k / len(y_true)) * (y_k - y_bar) ** 2
    return {
        "brier": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(y_bar * (1 - y_bar)),
    }


def compute_error_auroc(
    p_mean: np.ndarray,
    y_true: np.ndarray,
    sigma2_ep: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """AUROC de detección de errores: ¿predice σ²_ep cuándo el modelo falla?"""
    errors = ((p_mean >= threshold).astype(int) != y_true.astype(int)).astype(int)
    if errors.sum() == 0 or errors.sum() == len(errors):
        logging.warning(
            "compute_error_auroc: caso trivial (todos aciertos o todos errores)"
        )
        return 0.5
    return float(roc_auc_score(errors, sigma2_ep))

```


```python
# gamma_hadron_metrics se define más adelante en la sección Q-FACTOR

```


```python
def compute_bayesian_stats(
    all_preds: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calcula las cuatro cantidades bayesianas formales desde los pases MC crudos.

    Parameters
    ----------
    all_preds : (T, N) float32 — predicciones sigmoides de T pases MC

    Returns
    -------
    p_hat     : (N,) — p̂_γ   = E_T[p̂_t]
    sigma2_ep : (N,) — σ²_ep  = Var_T(p̂_t)
    H         : (N,) — H[y|x] = -p̂_γ log p̂_γ - (1-p̂_γ) log(1-p̂_γ)
    MI        : (N,) — MI[y,ω|x] = H[y|x] - E_T[H[y|x,ω_t]]
    """
    eps = 1e-8
    # ── p̂_γ — predicción principal ────────────────────────────────────────────
    p_hat = all_preds.mean(axis=0)  # (N,)

    # ── σ²_ep — varianza empírica entre pases MC ──────────────────────────────
    sigma2_ep = all_preds.var(axis=0)  # (N,)

    # ── H[y|x] — entropía predictiva total ────────────────────────────────────
    H = -(p_hat * np.log(p_hat + eps) + (1 - p_hat) * np.log(1 - p_hat + eps))  # (N,)

    # ── H[y|x, ω_t] — entropía por pase individual ────────────────────────────
    H_t = -(
        all_preds * np.log(all_preds + eps)
        + (1 - all_preds) * np.log(1 - all_preds + eps)
    )  # (T, N)

    # ── MI[y, ω | x] — información mutua (incertidumbre epistémica pura) ──────
    MI = H - H_t.mean(axis=0)  # (N,)

    return {
        "p_hat": p_hat,
        "sigma2_ep": sigma2_ep,
        "H": H,
        "MI": MI,
    }

```


```python
def mc_predict(
    model: tf.keras.Model,
    x_seq: np.ndarray,
    x_global: np.ndarray,
    T: int = 100,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    N = len(x_seq)
    batch_starts = range(0, N, batch_size)
    all_preds = np.empty((T, N), dtype=np.float32)

    for t in range(T):
        chunks = []
        for s in batch_starts:
            raw_output = model(
                [x_seq[s : s + batch_size], x_global[s : s + batch_size]],
                training=False,
            )
            # Manejo robusto: output único (tensor) o múltiple (dict/lista)
            if isinstance(raw_output, dict):
                pred = raw_output["particle_output"]
            elif isinstance(raw_output, (list, tuple)):
                pred = raw_output[0]  # particle_output es siempre el primero
            else:
                pred = raw_output  # output único — caso normal de training
            chunks.append(pred.numpy().reshape(-1))
        all_preds[t] = np.concatenate(chunks)

    stats = compute_bayesian_stats(all_preds)
    stats["all_preds"] = all_preds
    return stats
```


```python
class MCDropout(tf.keras.layers.Dropout):
    """
    Capa de Dropout personalizada para Monte Carlo (MC) Dropout.

    Mantiene el Dropout siempre activo, incluso durante la fase de inferencia
    (predicción). Esto permite realizar muestreo estocástico múltiple para estimar la incertidumbre epistémica del modelo, sin alterar el comportamiento de otras capas sensibles a la fase de ejecución (como BatchNormalization o LayerNormalization).
    """

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el Dropout a las entradas de la capa.

        Args:
            inputs (tf.Tensor): Tensor de entrada que pasará por la capa.
            training (Optional[bool], opcional): Flag booleano del modelo que indica si la red está en fase de entrenamiento o inferencia. Se ignora intencionalmente en esta clase. Por defecto es None.

        Returns:
            tf.Tensor: El tensor con el Dropout aplicado (siempre activo),
            manteniendo la misma dimensionalidad que `inputs`.
        """
        # Ignora el flag global 'training' y lo fuerza a True permanentemente
        return super().call(inputs, training=True)

    def get_config(self) -> Dict[str, Any]:
        """
        Devuelve la configuración de la capa para permitir la serialización
        y guardado correcto del modelo en Keras.

        Returns:
            Dict[str, Any]: Diccionario con los parámetros de inicialización de la capa.
        """
        return super().get_config()

```


```python
def transformer_block(
    x: tf.Tensor,
    prefix: str,
    dropout_rate: float = 0.10,
    spatial: bool = False,  # ← NUEVO: True si input viene de Conv1D
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Pre-LN Transformer block con MC Dropout en dos posiciones.

    Parameters
    ----------
    spatial : bool
        True  → usa MCSpatialDropout1D (input con correlaciones CNN).
        False → usa MCDropout estándar  (input RAW, sin correlaciones Conv1D).
    """
    # Selección de clase de dropout según origen del tensor
    DropLayer = MCSpatialDropout1D if spatial else MCDropout

    x_norm = layers.LayerNormalization(name=f"pre_mha_norm_{prefix}")(x)

    mha_layer = layers.MultiHeadAttention(num_heads=1, key_dim=32, name=f"mha_{prefix}")
    attn_output, attn_scores = mha_layer(x_norm, x_norm, return_attention_scores=True)

    # Dropout 1: sobre output de atención — usa tipo correcto según origen
    attn_output = DropLayer(dropout_rate, name=f"mc_drop_attn_{prefix}")(attn_output)

    recorder = layers.Lambda(lambda s: s, name=f"attention_scores_{prefix}")(
        attn_scores
    )

    x_add = layers.Add(name=f"add_{prefix}")([x, attn_output])
    x_norm_ffn = layers.LayerNormalization(name=f"pre_ffn_norm_{prefix}")(x_add)

    ffn = layers.Dense(128, activation="elu", name=f"ffn_dense_{prefix}1")(x_norm_ffn)

    # Dropout 2: dentro del FFN — mismo tipo
    ffn = DropLayer(dropout_rate, name=f"mc_drop_ffn_{prefix}")(ffn)

    ffn = layers.Dense(x.shape[-1], name=f"ffn_dense_{prefix}2")(ffn)
    x_out = layers.Add(name=f"ffn_add_{prefix}")([x_add, ffn])

    pooled = layers.GlobalAveragePooling1D(name=f"temporal_pooling_{prefix}")(x_out)

    return pooled, recorder

```


```python
def compute_file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Compute the cryptographic hash of a file for integrity verification.

    Reads the file in chunks to avoid loading large files (e.g. the processed_all_data.pkl dataset) entirely into memory.

    Parameters
    ----------
    path : Path
        Path to the file to hash.
    algorithm : str, optional
        Hash algorithm supported by hashlib (e.g. 'md5', 'sha256'). Use 'md5' for speed on large files; 'sha256' for higher collision resistance. Default is 'md5'.

    Returns
    -------
    str
        Lowercase hexadecimal hash string of the file contents.
    """
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

```


```python
def verify_dataset_integrity(
    dataset_path: Path,
    known_versions: dict[str, str],
) -> tuple[bool, str | None]:
    """
    Verify dataset integrity by matching its hash against known versions.

    Computes the MD5 hash of the dataset file and performs a reverse lookup in the known_versions registry to identify which registered version it corresponds to, if any. This guards against silent dataset changes (e.g. updates to the Google Drive source file) that would produce different results with the same code.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset file to verify (e.g. processed_all_data.pkl).
    known_versions : dict[str, str]
        Registry mapping descriptive version names to their expected MD5
        hashes. Example:
        {"v1_300_500_800GeV_angle60": "cf21f0de988a269fb5128e0e8993f632"}.

    Returns
    -------
    tuple[bool, str | None]
        - (True, version_name) if the file hash matches a known version.
        - (False, None) if the hash is not found in the registry, indicating either a new unregistered dataset version or file corruption.
    """
    actual_hash = compute_file_hash(dataset_path)

    # Búsqueda inversa: hash calculado → nombre de versión
    version_found = next(
        (
            name
            for name, expected_hash in known_versions.items()
            if actual_hash == expected_hash
        ),
        None,
    )

    if version_found is None:
        logging.warning(
            "Dataset hash '%s' not found in known versions registry. "
            "The dataset may have changed or is a new unregistered version. "
            "Known versions: %s",
            actual_hash,
            list(known_versions.keys()),
        )
        return False, None
    logging.info(
        "Dataset integrity verified — version: '%s', hash: %s",
        version_found,
        actual_hash,
    )
    return True, version_found
```


```python
class F1Score(tf.keras.metrics.Metric):
    """
    Calcula F1-Score para clasificación binaria a nivel de batch.
    """

    def __init__(
        self,
        name: str = "f1_score",
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:

        super().__init__(name=name, **kwargs)

        self.threshold = threshold

        # Usamos cast() para obligar al linter a ver el tipo correcto
        self.tp: tf.Variable = cast(
            tf.Variable,
            self.add_weight(
                name="tp",
                initializer="zeros",
                dtype=tf.float32,
            ),
        )

        self.fp: tf.Variable = cast(
            tf.Variable,
            self.add_weight(
                name="fp",
                initializer="zeros",
                dtype=tf.float32,
            ),
        )

        self.fn: tf.Variable = cast(
            tf.Variable,
            self.add_weight(
                name="fn",
                initializer="zeros",
                dtype=tf.float32,
            ),
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Any = None,
    ) -> None:

        # Convertimos probabilidades a clases binarias (usando >=)
        y_pred = tf.cast(
            y_pred >= self.threshold,
            tf.float32,
        )

        y_true = tf.cast(
            y_true,
            tf.float32,
        )

        tp = tf.reduce_sum(y_true * y_pred)

        fp = tf.reduce_sum((1 - y_true) * y_pred)

        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self) -> tf.Tensor:

        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())

        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

        return (
            2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        )

    def reset_state(self) -> None:

        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self) -> dict[str, Any]:

        config = super().get_config()

        config.update(
            {
                "threshold": self.threshold,
            }
        )

        return config

```


```python
# ============================================================
# Q-FACTOR and efficiencies
def gamma_hadron_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute standard gamma/hadron discrimination metrics for IACTs.

    Implements the figures of merit used in ground-based gamma-ray astronomy (MAGIC, CTA, HAWC) to evaluate the quality of a gamma/hadron classifier at a given decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels. Convention: 1 = gamma, 0 = proton/hadron.
    y_pred_proba : np.ndarray
        Predicted probability of the gamma class, in [0, 1].
    threshold : float, optional
        Decision threshold for converting probabilities to binary labels.
        Default is 0.5.

    Returns
    -------
    dict with keys:
        efficiency_gamma : float
            True positive rate for gamma events (signal efficiency). epsilon_gamma = TP / (TP + FN).
        contamination_hadron : float
            False positive rate for hadron events (background leakage). epsilon_hadron = FP / (FP + TN).
        q_factor : float
            Figure of merit Q = epsilon_gamma / sqrt(epsilon_hadron). Proportional to the improvement in source detection sensitivity over an uncut dataset. A value of Q=5 reduces required observation time by a factor of 25 (t  1/Q²).
        purity_gamma : float
            Fraction of selected events that are true gammas. purity = TP / (TP + FP).
        significance_lima : float
            Li & Ma significance (ApJ 272:317, 1983) computed with alpha=1.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    # Assuming 1=gamma, 0=proton
    gamma_mask = y_true == 1
    proton_mask = y_true == 0
    # True Positives, False Positives, etc.
    tp = np.sum((y_pred == 1) & gamma_mask)
    fp = np.sum((y_pred == 1) & proton_mask)
    fn = np.sum((y_pred == 0) & gamma_mask)
    tn = np.sum((y_pred == 0) & proton_mask)
    epsilon_gamma = tp / (tp + fn) if (tp + fn) > 0 else 0  # Gamma efficiency
    epsilon_hadron = fp / (fp + tn) if (fp + tn) > 0 else 0  # Hadronic contamination
    # Q-factor
    q_factor = epsilon_gamma / np.sqrt(epsilon_hadron) if epsilon_hadron > 0 else 0
    # Gamma sample purity
    purity = tp / (tp + fp) if (tp + fp) > 0 else 0
    return {
        "efficiency_gamma": epsilon_gamma,
        "contamination_hadron": epsilon_hadron,
        "q_factor": q_factor,
        "purity_gamma": purity,
        "significance_lima": calculate_lima_significance(tp, fp),
    }


def calculate_lima_significance(
    n_on: float,
    n_off: float,
    alpha: float = 1.0,
) -> float:
    """
    Compute the Li & Ma significance for gamma-ray source detection.
    Implements Equation 17 from Li & Ma (1983, ApJ 272:317-324), the statistical standard for claiming discovery of an astrophysical gamma-ray source. A significance of 5 corresponds to a random fluctuation probability of 2.8 × 10.

    Parameters
    ----------
        n_on : float
            Number of events detected in the source (ON) region.
        n_off : float
            Number of events detected in the background (OFF) control region.
        alpha : float, optional
            Exposure ratio between ON and OFF regions (t_on / t_off).
            Default is 1.0 (equal exposure).

    Returns
    -------
    float
        Li & Ma significance in units of sigma ().
        Returns 0.0 if n_on or n_off is zero.
    Notes
    -----
    The discovery threshold in gamma-ray astronomy is S > 5. For a balanced test set (alpha=1), n_on and n_off correspond to true positives and false positives respectively.
    """
    if n_on == 0 or n_off == 0:
        return 0.0
    n_on = float(n_on)
    n_off = float(n_off)
    term1 = n_on * np.log((1 + alpha) / alpha * (n_on / (n_on + n_off)))
    term2 = n_off * np.log((1 + alpha) * (n_off / (n_on + n_off)))
    significance = np.sqrt(2 * (term1 + term2))
    return significance

```

### Reproducibility and Environment

- Fixed seeds (numpy, random, tensorflow).
- GPU memory growth + mixed precision.
- Paths and artifact directories.



```python
# ----------------------------- #
# Reproducibility configuration #
# ----------------------------- #
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
logging.info("Random seeds fixed — SEED=%d", SEED)
DEBUG_MODE = True  # True = usa subconjunto pequeño para pruebas rápidas
DEBUG_PERCENTAGE = 8  # porcentaje de eventos a usar en modo debug

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

gpus = tf.config.list_physical_devices("GPU")

if not gpus:
    logging.warning(
        "No GPU detected — training will run on CPU and may be significantly slower"
    )
else:
    logging.info("GPUs detected: %s", [g.name for g in gpus])

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("Memory growth enabled for %s", gpu.name)
        except RuntimeError as e:
            # RuntimeError ocurre cuando la GPU ya fue inicializada antes
            # de llamar set_memory_growth — no es fatal pero debe registrarse
            logging.warning("Could not enable memory growth for %s: %s", gpu.name, e)
        except Exception as e:
            # Cualquier otro fallo inesperado de driver o configuración
            logging.error("Unexpected error configuring %s: %s", gpu.name, e)

    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logging.info(
            "Mixed precision enabled (float16) — faster training on compatible GPUs"
        )
    except Exception as e:
        logging.warning(
            "Could not enable mixed precision, falling back to float32: %s", e
        )
```

### Data Source and Physics Cuts

- Monte Carlo EAS simulations for CONDOR.
- Energy bands: 3E2, 5E2, 8E2; zenith ≤ 40°; min total particles ≥ 30.
- Cached pickle download if missing.



```python
# * Se configura archivo de log con timestamp para cada ejecución, guardando tanto en consola como en un archivo dentro de pipeline_artifacts.
log_filename = (
    f"pipeline_artifacts/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
    force=True,
)
```


```python
# ----------------------------- #
# Paths and pipeline parameters #
# ----------------------------- #
BASE_DIR: Path = Path.cwd()
CACHE_FILE = BASE_DIR / "processed_all_data.pkl"

DATASET_VERSIONS: dict[str, str] = {
    "v1_(300_500_800GeV_angle60)": "cf21f0de988a269fb5128e0e8993f632",
    # "v2_(300_500_800GeV_1TeV_angle60)": "...",
}

if not CACHE_FILE.exists():
    url = "https://drive.google.com/uc?id=1Z6EqKXUzKrlxIYcbYg1bPfNkNhfP6Pgz"
    gdown.download(url, str(CACHE_FILE), quiet=False)
is_valid, dataset_version = verify_dataset_integrity(
    CACHE_FILE,
    DATASET_VERSIONS,
)
ARTIFACTS_DIR: Path = BASE_DIR / "pipeline_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT_NAME = "CONDOR_EAS_Reconstruction"
mlflow_tracking_uri: Path = ARTIFACTS_DIR / "mlruns"
mlflow_tracking_uri.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = mlflow_tracking_uri.resolve().as_uri()

ENERGY_FILTER = ("3E2", "5E2", "8E2")
ANGLE_MAX = 40.0  # Oficial (al 04 de Mayo del 2026) es 40.
MIN_TOTAL_PARTICLES = 30  # Oficial (al 04 de Mayo del 2026) son 30.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test will be 0.15 (remaining)
BATCH_SIZE = 32
EPOCHS = 30
MC_T = 100
MC_BATCH = 64
sns.set_context("poster")
sns.set_style("ticks")


logging.info("Model saved at: %s", BASE_DIR)
logging.info("Cache file: %s", CACHE_FILE)
logging.info("Artifacts directory: %s", ARTIFACTS_DIR)
```


```python
# / ── Metadata de configuración inicial ──────────────────────────────────────────
metadata = {
    "run_timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "debug_mode": DEBUG_MODE,
    "filters": {
        "energies_used": list(ENERGY_FILTER),
        "angle_max_deg": ANGLE_MAX,
        "min_total_particles": MIN_TOTAL_PARTICLES,
    },
}
logging.info("Initial configuration: %s", json.dumps(metadata, indent=2))
```


```python
# Configuración inicial
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Al asignar set_experiment a una variable, obtenemos sus metadatos (incluyendo su ID)
experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# 1. Consultamos si existe alguna sesión activa "zombie" en memoria
active_run = mlflow.active_run()

if active_run:
    active_run_id = active_run.info.run_id
    active_exp_id = active_run.info.experiment_id
    active_run_name = active_run.data.tags.get("mlflow.runName", "Sin_Nombre")

    # 2. Comprobamos si el run activo pertenece a nuestro experimento actual
    if active_exp_id == experiment.experiment_id:
        logging.warning(
            "Se detectó un run activo previo en este experimento "
            "(Nombre: %s | ID: %s). "
            "Se cerrará automáticamente para evitar conflictos.",
            active_run_name,
            active_run_id,
        )
    else:
        logging.warning(
            "Se detectó un run activo de un experimento distinto "
            "(Exp ID: %s | Run ID: %s). Cerrándolo por seguridad.",
            active_exp_id,
            active_run_id,
        )

    # 3. Cerramos el run activo para liberar el bloqueo
    mlflow.end_run()

# 4. Ahora es 100% seguro iniciar el nuevo run
run_name = f"cnn_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow_run = mlflow.start_run(run_name=run_name)

logging.info(
    "Nuevo run de MLflow iniciado con éxito: %s (ID: %s)",
    run_name,
    mlflow_run.info.run_id,
)
mlflow.set_tag("researcher", "Alan")  # ¿Quién corrió esto?
mlflow.set_tag("stage", "exploracion")  # ¿Es prueba o es el final?
mlflow.set_tag("data_balance", "stratified")  # Técnica usada
mlflow.set_tag("model_architecture", "CNN+Trans")  # Resumen rápido
```


```python
# ----------------------------- #
# Load from pickle              #
# ----------------------------- #
def load_dataset_from_cache(cache_path: Path) -> pd.DataFrame:
    """
    Load the preprocessed dataset from a pickle cache file.

    Parameters
    ----------
    cache_path : Path
        Path to the pickle file containing the preprocessed dataset.
        Must contain the columns: shower_data, angle, label, energy,
        total_particles, max_time.

    Returns
    -------
    pd.DataFrame
        DataFrame with the full dataset, index reset.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist at the given path.
    ValueError
        If the pickle contains an unsupported type or is missing
        required columns.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Regenerate the pickle before running the pipeline."
        )
    with cache_path.open("rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported cache format.")
    required_cols = {
        "shower_data",
        "angle",
        "label",
        "energy",
        "total_particles",
        "max_time",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError("The pickle does not contain all required columns.")
    return df.reset_index(drop=True)


df_full = load_dataset_from_cache(CACHE_FILE)

df_full = df_full[
    (df_full["energy"].isin(ENERGY_FILTER))
    & (df_full["angle"] <= ANGLE_MAX)
    & (df_full["total_particles"] >= MIN_TOTAL_PARTICLES)
].reset_index(drop=True)
logging.info(
    "Dataset loaded and filtered: %d events remain after applying energy, angle, and particle count filters.",
    len(df_full),
)
if df_full.empty:
    raise RuntimeError("No events found after filtering by energy / angle / particles.")

if DEBUG_MODE:
    df_full = df_full.sample(
        n=min(len(df_full) * DEBUG_PERCENTAGE // 100, len(df_full)),
        random_state=SEED,
    ).reset_index(drop=True)
    logging.warning(
        "DEBUG_MODE active — using %d events out of full dataset. "
        "Results are NOT representative.",
        len(df_full),
    )

df_full["idx"] = df_full.index
```


```python
df_full.head()
```


```python
df_full.at[1, "shower_data"]
```

### Description of CONDOR Dataset Variables

Below are the variables used in this notebook, originating from simulated extensive air showers (EAS) under the CONDOR observatory conditions. These variables are divided into sequence features (individual hits), global event features, and target variables.

| Variable                  | Description                                                                                                    |
| :------------------------ | :------------------------------------------------------------------------------------------------------------- |
| **`shower_data`**         | Raw sequence of hits recorded by the detectors. This is the main input for the convolutional/recurrent branch. |
| `detector_id`             | Unique identifier of the detector that registered the signal.                                                  |
| `x_center`, `y_center`    | Spatial coordinates (in meters) of the center of the activated detector.                                       |
| `t_bin`                   | Discretized time bin when the signal was recorded in the detector.                                             |
| `particle_count`          | Number of secondary particles detected in the detector at that instant.                                        |
| `total_energy` (hit)      | Energy deposited in the specific detector during the hit.                                                      |
| **`total_particles`**     | Total sum of particles detected across the entire array (global feature).                                      |
| **`active_detectors`**    | Number of unique detectors activated during the event (global feature).                                        |
| **`max_time / duration`** | Total temporal duration of the event in the array (global feature).                                            |
| **`energy_central`**      | Accumulated energy specifically in the 16 central detectors of the array (global feature).                     |
| **`label`**               | Classification label: Type of primary particle (Photon $\gamma$ vs Proton/Hadron).                             |
| **`angle`**               | Zenith angle of the incoming primary particle (in degrees).                                                    |
| **`energy`**              | Energy of the primary particle (in GeV).                                                                       |

> **Note:** Sequence variables (`detector_id`, `t_bin`, etc.) are processed as time series, while global variables (`total_particles`, etc.) are injected directly into the dense layers of the hybrid model.


### Detector Geometry Reconstruction

- Map detector_id → (x_center, y_center).
- Save catalog for spatial features.



```python
# ----------------------------- #
# Catálogo de detectores        #
# ----------------------------- #
def build_detector_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a catalog of unique detectors with their spatial coordinates.

    Extracts detector_id, x_center, and y_center from the raw shower_data
    sequences and deduplicates by detector_id, producing a reference table
    of the physical array geometry.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a shower_data column where each entry is an
        array of shape (n_hits, n_raw_features). Columns 0, 5, and 6
        correspond to detector_id, x_center, and y_center respectively.

    Returns
    -------
    pd.DataFrame
        Catalog with columns [detector_id, x_center, y_center], one row
        per unique detector, sorted by detector_id. Returns an empty
        DataFrame if no valid sequences are found.
    """
    positions = []
    for seq in df["shower_data"]:
        arr = np.asarray(seq, dtype=np.float32)
        if arr.size == 0:
            continue
        positions.append(arr[:, [0, 5, 6]])  # detector_id, x_center, y_center
    if not positions:
        return pd.DataFrame(columns=["detector_id", "x_center", "y_center"])
    catalog = (
        pd.DataFrame(
            np.vstack(positions), columns=["detector_id", "x_center", "y_center"]
        )
        .drop_duplicates(subset=["detector_id"])
        .sort_values("detector_id")
        .reset_index(drop=True)
    )
    catalog["detector_id"] = catalog["detector_id"].astype(int)
    return catalog


detector_catalog = build_detector_catalog(df_full)
detector_catalog.to_csv(ARTIFACTS_DIR / "detector_catalog.csv", index=False)
```


```python
detector_catalog
```

### Balancing Strategy

- Undersample/oversample by (label, angle, energy) to remove spectral/geom biases.
- Target = median group size.



```python
def balance_by_group(
    df: pd.DataFrame, group_cols, random_state: int
) -> tuple[pd.DataFrame, int]:
    """
    Balance the dataset by resampling each group to the median group size.

    Groups are defined by the combination of values in group_cols. Groups
    smaller than the target are oversampled with replacement; groups larger
    than the target are undersampled without replacement. This strategy
    removes spectral and geometric biases introduced by unequal simulation
    statistics across (label, angle, energy) combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to balance.
    group_cols : sequence of str
        Column names used to define groups (e.g. ['label', 'angle', 'energy']).
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[pd.DataFrame, int]
        - Balanced and shuffled DataFrame.
        - Target sample size per group (median of original group sizes).

    Raises
    ------
    RuntimeError
        If no non-empty groups are found after grouping.
    """
    counts = df.groupby(list(group_cols)).size()
    counts = counts[counts > 0]
    if counts.empty:
        raise RuntimeError("No groups available for balancing.")
    target = int(counts.median())
    balanced_parts = []
    for _, group in df.groupby(list(group_cols)):
        replace = len(group) < target
        balanced_parts.append(
            group.sample(n=target, replace=replace, random_state=random_state)
        )
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    return balanced_df, target


df_balanced, target_per_group = balance_by_group(
    df_full,
    group_cols=("label", "angle", "energy"),
    random_state=SEED,
)
```


```python
# / ── Metadata post balanceo ─────────────────────────────────────────────────────
metadata["dataset"] = {
    "balance_target_per_group": target_per_group,
    "num_events_total": int(len(df_balanced)),
    "cache_file": str(CACHE_FILE),
    "hash_md5": compute_file_hash(CACHE_FILE),
    "version": dataset_version,
    "hash_verified": is_valid,
}
logging.info(
    "Dataset balanced: %d events total, target %d per group. Version: %s, Hash valid: %s",
    len(df_balanced),
    target_per_group,
    dataset_version,
    is_valid,
)
```


```python
display(df_balanced)
```

### Distribution Checks

- Histograms for label, angle, energy (pre/post balance).
- Cross-table summaries.



```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Before balancing
df_full_counts = {
    "label": df_full["label"].value_counts().sort_index(),
    "angle": df_full["angle"].value_counts().sort_index(),
    "energy": df_full["energy"].value_counts().sort_index(),
}

# After balancing
df_balanced_counts = {
    "label": df_balanced["label"].value_counts().sort_index(),
    "angle": df_balanced["angle"].value_counts().sort_index(),
    "energy": df_balanced["energy"].value_counts().sort_index(),
}

features = ["label", "angle", "energy"]
titles = ["Label", "Angle", "Energy"]

for i, (feature, title) in enumerate(zip(features, titles)):
    ax = axes[i]
    ax.bar(
        df_full_counts[feature].index,
        df_full_counts[feature].values,
        alpha=0.6,
        label="Before",
    )
    ax.bar(
        df_balanced_counts[feature].index,
        df_balanced_counts[feature].values,
        alpha=0.6,
        label="After",
    )
    ax.set_title(f"{title} Distribution")
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.show()
mlflow.log_figure(fig, "graficos/distribution_summary.png")
```


```python
def balance_summary(df_pre: pd.DataFrame, df_post: pd.DataFrame) -> None:
    """
    Display a cross-tabulation of event counts before and after balancing.

    Groups events by (energy, label, angle) and pivots angle as columns, producing a side-by-side comparison of the dataset distribution before and after the balancing step. Output is rendered via IPython display.

    Parameters
    ----------
    df_pre : pd.DataFrame
        Original unbalanced dataset. Must contain columns: label, energy, angle.
    df_post : pd.DataFrame
        Balanced dataset returned by balance_by_group(). Must contain the same columns as df_pre.

    Returns
    -------
    None
        Results are rendered inline via display(). No value is returned.
    """
    label_map = {0: "Photon", 1: "Proton"}

    def agregar_tabla(df, nombre):
        tabla = (
            df.assign(label=df["label"].map(label_map))
            .groupby(["energy", "label", "angle"])
            .size()
            .unstack("angle", fill_value=0)
        )
        tabla["Total"] = tabla.sum(axis=1)
        tabla["Set"] = nombre
        return tabla.reset_index()

    tabla_pre = agregar_tabla(df_pre, "Before")
    tabla_post = agregar_tabla(df_post, "After")
    tabla_final = pd.concat([tabla_pre, tabla_post], ignore_index=True)
    display(tabla_final)


balance_summary(df_full, df_balanced)

```

### Sequence Features

- Time-ordered hits: detector_id, particle_count, t_bin, total_energy, x_center, y_center.
- Sorted by t_bin; padded sequences.



```python
RAW_FEATURE_NAMES = [
    "detector_id",  # 0
    "t_bin",  # 1
    "particle_count",  # 2
    "mean_x",  # 3
    "mean_y",  # 4
    "x_center",  # 5
    "y_center",  # 6
    "total_energy",  # 7
    "percentage_in_condor",  # 8
]

FEATURE_NAMES = [
    "detector_id",
    "particle_count",
    "t_bin",
    "total_energy",
    "x_center",
    "y_center",
]

# Si el dataset cambia de orden pero mantiene los nombres, esto jamás se romperá.
FEATURE_ORDER = [RAW_FEATURE_NAMES.index(name) for name in FEATURE_NAMES]
T_BIN_IDX = RAW_FEATURE_NAMES.index(
    "t_bin"
)  # Se necesita la posición de t_bin para ordenar


def build_sequence_list(df: pd.DataFrame) -> list[np.ndarray]:
    """
    Extract and sort hit sequences from raw shower_data by arrival time.

    Each event is represented as a variable-length array of detector hits. Hits are reordered by t_bin in ascending order and projected onto the feature subset defined by FEATURE_NAMES.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a shower_data column where each entry is an array of shape (n_hits, n_raw_features).

    Returns
    -------
    list[np.ndarray]
        List of arrays, one per event, each of shape (n_hits, len(FEATURE_NAMES)) and dtype float32. Empty events are represented as zero-row arrays.
    """
    sequences = []

    for seq in df["shower_data"]:
        arr = np.asarray(seq, dtype=np.float32)

        if arr.size == 0:
            sequences.append(np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32))
            continue

        order = np.argsort(arr[:, T_BIN_IDX])

        ordered = arr[order][:, FEATURE_ORDER].astype(np.float32)
        sequences.append(ordered)

    return sequences


# Ejecución de la extracción de secuencias y variables objetivo
X_sequences = build_sequence_list(df_balanced)
angles = df_balanced["angle"].to_numpy(dtype=np.float32)
labels = df_balanced["label"].to_numpy(dtype=np.int32)
energies = df_balanced["energy"].to_numpy()

```


```python
display(X_sequences)

```

### Global Features

- Total particles, total energy, active detectors, duration, central energy (inner detectors).
- Central IDs: 16 central detectors.



```python
detector_catalog["distance"] = np.hypot(
    detector_catalog["x_center"], detector_catalog["y_center"]
)
central_ids = (
    detector_catalog.nsmallest(16, "distance")["detector_id"].astype(int).tolist()
)


def compute_global_features(
    sequences: list[np.ndarray], central_detectors: list[int]
) -> np.ndarray:
    """
    Compute scalar global features summarizing each EAS event.

    Global features capture array-level statistics that complement the hit-level sequence representation. The 16 central detectors are used to compute a core energy proxy that is particularly sensitive to shower maximum depth and primary particle type.

    Parameters
    ----------
    sequences : list[np.ndarray]
        List of hit sequences, each of shape (n_hits, 6) with columns [detector_id, particle_count, t_bin, total_energy, x_center, y_center].
    central_detectors : list[int]
        IDs of the detectors closest to the array center, used to compute the central energy feature.

    Returns
    -------
    np.ndarray
        Array of shape (n_events, 5) and dtype float32. Column definitions:

        - [0] total_particles  : sum of particle counts across all hits.
        - [1] total_energy     : sum of energy deposited across all hits.
        - [2] active_detectors : number of unique detectors that recorded signal.
        - [3] duration         : difference between maximum and minimum t_bin.
        - [4] energy_central   : sum of energy deposited in the central detectors.

    Notes
    -----
    Events with empty sequences (size == 0) are assigned a zero vector. The central energy feature (column 4) is zero when no central detector recorded signal in a given event.
    """
    central_detectors = np.array(central_detectors, dtype=np.int32)
    features = np.zeros((len(sequences), 5), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        if seq.size == 0:
            continue
        det_ids = seq[:, 0].astype(np.int32)
        particle_counts = seq[:, 1]
        t_bins = seq[:, 2]
        tot_energy = seq[:, 3]

        total_particles = float(particle_counts.sum())
        total_energy = float(tot_energy.sum())
        active_detectors = float(np.unique(det_ids).size)
        duration = float(t_bins.max() - t_bins.min())
        mask_central = np.isin(det_ids, central_detectors)  # En esta linea se revisa
        energy_central = (
            float(tot_energy[mask_central].sum()) if np.any(mask_central) else 0.0
        )

        features[idx] = [
            total_particles,
            total_energy,
            active_detectors,
            duration,
            energy_central,
        ]
    return features


X_global = compute_global_features(X_sequences, central_ids)
X_padded = pad_sequences(X_sequences, padding="post", dtype="float32")
max_sequence_length = int(X_padded.shape[1])
```


```python
X_global[0]
```

### Stratified Split

- 70/15/15 stratified on (angle, label).
- Energy both categorical (bins) and continuous (scaled).
- Record means/std for scaling.



```python
indices = np.arange(len(X_padded))
strata = np.array([f"{int(round(a))}_{lbl}" for a, lbl in zip(angles, labels)])

train_idx, temp_idx = train_test_split(
    indices,
    test_size=1.0 - TRAIN_RATIO,
    stratify=strata,
    random_state=SEED,
    shuffle=True,
)

val_fraction = VAL_RATIO / (1.0 - TRAIN_RATIO)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=1.0 - val_fraction,
    stratify=strata[temp_idx],
    random_state=SEED,
    shuffle=True,
)


def split_arrays(
    arr: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Index an array by train, validation, and test index sets.

    Convenience wrapper to apply the same three-way index split to any array along its first axis. Intended to be called once per array (sequences, global features, labels) after computing the stratified split indices.

    Parameters
    ----------
    arr : np.ndarray
        Array to split. Shape (n_events, ...).
    tr : np.ndarray
        Integer indices for the training set.
    va : np.ndarray
        Integer indices for the validation set.
    te : np.ndarray
        Integer indices for the test set.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray] (train_arr, val_arr, test_arr), each a view of the original array indexed by the corresponding index set.
    """
    return arr[tr], arr[va], arr[te]


X_train, X_val, X_test = split_arrays(X_padded, train_idx, val_idx, test_idx)
y_energy_train, y_energy_val, y_energy_test = split_arrays(
    energies, train_idx, val_idx, test_idx
)
y_angle_train, y_angle_val, y_angle_test = split_arrays(
    angles, train_idx, val_idx, test_idx
)
Xg_train, Xg_val, Xg_test = split_arrays(X_global, train_idx, val_idx, test_idx)
y_label_train, y_label_val, y_label_test = split_arrays(
    labels, train_idx, val_idx, test_idx
)

energy_levels = sorted(
    {e for e in df_balanced["energy"]}, key=lambda x: float(x.replace("E", "e"))
)
```


```python
# / ── Metadata post split ────────────────────────────────────────────────────────
metadata["train_val_test_split"] = {
    "train": int(len(train_idx)),
    "val": int(len(val_idx)),
    "test": int(len(test_idx)),
}
metadata["features"] = {
    "feature_order": FEATURE_NAMES,
    "max_sequence_length": max_sequence_length,
    "global_features_definition": [
        "sum_particle_count",
        "sum_total_energy",
        "num_active_detectors",
        "duration_tbin",
        "sum_total_energy_central_ids",
    ],
    "central_detector_ids": [int(x) for x in central_ids],
}

# Primera escritura a disco — si el entrenamiento falla, esto ya está guardado
metadata_path = ARTIFACTS_DIR / "preprocessing_metadata.json"
with metadata_path.open("w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2)
logging.info("Metadata (pre-training) saved to: %s", metadata_path)

logging.info(
    "Metadata post initial configuration, balancing and split saved to: %s",
    metadata_path,
)
```


```python
datasets = [
    ("Train", y_label_train),
    ("Validation", y_label_val),
    ("Test", y_label_test),
]
order = [dataset_name for dataset_name, _ in datasets]
frames = []
for name, particle_arr in datasets:
    frames.append(
        pd.DataFrame(
            {
                "dataset": name,
                "particle": particle_arr.astype(int),
            }
        )
    )

dist_df = pd.concat(frames, ignore_index=True)

fig, ax = plt.subplots(figsize=(18, 5))

feature = "particle"
title = "Distribution by particle"

counts = (
    dist_df.groupby(["dataset", feature])
    .size()
    .unstack("dataset", fill_value=0)
    .sort_index()
)

counts = counts[order]

counts.plot(kind="bar", ax=ax)

ax.set_title(title)
ax.set_xlabel(feature.capitalize())
ax.set_ylabel("Events count")
ax.legend(title="Dataset")

plt.tight_layout()
plt.show()

mlflow.log_figure(fig, "graficos/distribution_by_objective_post_balancing.png")
```

### Hybrid CNN-Transformer with Multi-Task Heads

- Sequence branch: Conv1D stack → Transformer blocks (per head tap).
- Global branch: dense layers.
- Heads: particle (sigmoid), angle (linear), energy (linear, scaled).
- Attention taps for interpretability.



```python
model_dir = ARTIFACTS_DIR / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "condor_multitask_model.keras"
```


```python
def build_bayesian_classifier(
    sequence_shape: tuple[int, int],
    global_dim: int,
    dropout_cnn: float = 0.30,
    dropout_head: float = 0.40,
    dropout_transformer: float = 0.10,
    include_attention_outputs: bool = False,  # ← mismo patrón que el original
) -> tf.keras.Model:
    """
    include_attention_outputs=False → un output: particle_output (sigmoid).
                                      Usar para training, compile, mc_predict.
    include_attention_outputs=True  → tres outputs: particle_output +
                                      attn_class_cnn + attn_class_raw.
                                      Usar para XAI. Mismos pesos, mismo grafo.
    """
    seq_input = Input(shape=sequence_shape, name="seq_input")
    global_input = Input(shape=(global_dim,), name="global_input")

    x_raw = layers.Masking(mask_value=0.0, name="masking")(seq_input)

    # ── CNN branch ────────────────────────────────────────────────────────────
    x_cnn = layers.Conv1D(128, 7, padding="same", activation="elu", name="conv_0")(
        x_raw
    )
    x_cnn = layers.BatchNormalization(name="bn_0")(x_cnn)
    x_cnn = layers.Conv1D(128, 7, padding="same", activation="elu", name="conv_1")(
        x_cnn
    )
    x_cnn = layers.BatchNormalization(name="bn_1")(x_cnn)
    x_cnn = layers.Conv1D(128, 7, padding="same", activation="elu", name="conv_2")(
        x_cnn
    )
    x_cnn = layers.BatchNormalization(name="bn_2")(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2, padding="same", name="pool_2")(x_cnn)
    x_cnn = layers.Dense(128, activation="elu", name="dense_after_cnn1")(x_cnn)
    x_cnn = layers.Dense(256, activation="elu", name="dense_after_cnn2")(x_cnn)
    x_cnn = MCSpatialDropout1D(dropout_cnn, name="dropout_after_cnn")(x_cnn)
    x_cnn = layers.Dense(128, activation="elu", name="dense_after_cnn4")(x_cnn)

    # ── RAW branch ────────────────────────────────────────────────────────────
    x_raw_proj = layers.Dense(128, activation="elu", name="raw_projection")(x_raw)

    # ── Transformer blocks ────────────────────────────────────────────────────
    feat_cnn, attn_cnn = transformer_block(
        x_cnn, "class_cnn", dropout_transformer, spatial=True
    )
    feat_raw, attn_raw = transformer_block(
        x_raw_proj, "class_raw", dropout_transformer, spatial=False
    )

    # ── Fusión y cabeza ───────────────────────────────────────────────────────
    feat_fused = layers.Concatenate(name="feat_fused")([feat_cnn, feat_raw])
    merged = layers.Concatenate(name="merge_particle")([feat_fused, global_input])

    x = MCDropout(dropout_head, name="dropout_particle")(merged)
    x = layers.Dense(128, activation="elu", name="dense_particle")(x)
    x = MCDropout(dropout_head, name="dropout_particle_2")(x)
    particle_out = layers.Dense(
        1, activation="sigmoid", dtype="float32", name="particle_output"
    )(x)

    # ── Output según modo ─────────────────────────────────────────────────────
    if include_attention_outputs:
        outputs = {
            "particle_output": particle_out,
            "attn_class_cnn": attn_cnn,
            "attn_class_raw": attn_raw,
        }
    else:
        outputs = particle_out

    return tf.keras.Model(
        inputs=[seq_input, global_input],
        outputs=outputs,
        name="CONDOR_BayesianClassifier",
    )
```


```python
def compile_bayesian_model(
    model: tf.keras.Model,
    learning_rate: float = 5e-4,
    clipnorm: float = 1.0,
    loss_weight_particle: float = 1.0,
) -> tuple[tf.keras.Model, dict]:

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "particle_output": "binary_crossentropy",
        },
        loss_weights={
            "particle_output": loss_weight_particle,  # explícito aunque sea 1.0
        },
        metrics={
            "particle_output": [F1Score(name="f1_score"), "accuracy"],
        },
    )

    compile_config = {
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": float(learning_rate),
        "clipnorm": float(clipnorm),  # ← añadido
        "loss": "binary_crossentropy",
        "loss_weight_particle": float(loss_weight_particle),  # ← añadido
    }

    return model, compile_config
```


```python
model = build_bayesian_classifier(
    sequence_shape=(max_sequence_length, len(FEATURE_NAMES)),
    global_dim=Xg_train.shape[1],
    include_attention_outputs=False,
)
model, compile_config = compile_bayesian_model(model)

# Para XAI — mismos pesos, grafo compartido, outputs adicionales
attn_model = build_bayesian_classifier(
    sequence_shape=(max_sequence_length, len(FEATURE_NAMES)),
    global_dim=Xg_train.shape[1],
    include_attention_outputs=True,
)
attn_model.set_weights(model.get_weights())

```

### Training Protocol

- Losses: BCE (particle).
- Loss weights: particle 0.6.
- Optimizer: Adam lr=5e-4, clipnorm=1.0.
- Callbacks: EarlyStopping, ReduceLROnPlateau y ModelCheckpoint.
- Mixed precision if GPU.

### Model Fitting and Persistence

- Train with stratified splits.
- Save model, history, attention map.
- Store artifacts under pipeline_artifacts/.



```python
mlflow.log_params(
    {
        # Pipeline
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "debug_mode": DEBUG_MODE,
        "angle_max_deg": ANGLE_MAX,
        "dataset_version": dataset_version,  # viene de verify_dataset_integrity()
        # Modelo — vienen del retorno de compile_model()
        **compile_config,
    }
)

logging.info("MLflow run started — name: %s", run_name)
```


```python
# ----------------------------- #
# Training                      #
# ----------------------------- #
best_model_path = ARTIFACTS_DIR / "condor_best_model.keras"
logging.info(
    "Begin training — best model checkpoints will be saved to: %s", best_model_path
)

checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_loss",  # Métrica a observar (debe coincidir con EarlyStopping)
    save_best_only=True,  # Solo sobreescribe si la métrica mejora
    save_weights_only=False,  # False = guarda el modelo COMPLETO (arquitectura y pesos)
    mode="min",
    verbose=1,
)

earlystopping_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1,
)

reducelronplateau_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_delta=1e-4,
    verbose=1,
    min_lr=1e-6,
)

callbacks_list = [
    earlystopping_cb,
    reducelronplateau_cb,
    checkpoint_cb,
]


history = model.fit(
    x=[X_train, Xg_train],
    y=y_label_train,
    validation_data=([X_val, Xg_val], y_label_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

model.save(filepath=model_path)
logging.info("Model saved at: %s", model_path)

for epoch, (loss, val_loss) in enumerate(
    zip(
        history.history["loss"],
        history.history["val_loss"],
    )
):
    mlflow.log_metrics(
        {"train_loss": loss, "val_loss": val_loss},
        step=epoch,
    )
logging.info(
    "MLflow — training curves logged (%d epochs)",
    len(history.history["loss"]),
)
```

### Training Curves

- Losses per head (train/val).
- Metrics: F1 (particle), MAE (angle, energy).



```python
history_csv_path = model_dir / "training_history.csv"
plots_dir = ARTIFACTS_DIR / "diagnostics"
plots_dir.mkdir(parents=True, exist_ok=True)

if "history" not in globals() or history is None:
    if history_csv_path.exists():
        history_df = pd.read_csv(history_csv_path)

        class HistoryObj:
            def __init__(self, df):
                self.history = {col: df[col].values for col in df.columns}

        history = HistoryObj(history_df)
        logging.info(
            "[Training curves] Training history loaded from: %s", history_csv_path
        )
        # print(f"Training history loaded from: {history_csv_path}")
    else:
        raise FileNotFoundError(f"Training history not found at {history_csv_path}")

# Training curves — Bayesian Classifier (single output)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Panel 1: Loss
axes[0].plot(history.history["loss"], label="Train", color="tab:blue")
axes[0].plot(
    history.history["val_loss"], label="Validation", color="tab:orange", linestyle="--"
)
axes[0].set_title("Binary Crossentropy Loss")
axes[0].set_ylabel("Loss")
axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Panel 2: F1 Score
axes[1].plot(history.history["f1_score"], label="Train", color="tab:blue")
axes[1].plot(
    history.history["val_f1_score"],
    label="Validation",
    color="tab:orange",
    linestyle="--",
)
axes[1].set_title("F1 Score")
axes[1].set_ylabel("F1 Score")
axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3)
axes[1].legend()

# Panel 3: Accuracy
axes[2].plot(history.history["accuracy"], label="Train", color="tab:blue")
axes[2].plot(
    history.history["val_accuracy"],
    label="Validation",
    color="tab:orange",
    linestyle="--",
)
axes[2].set_title("Accuracy")
axes[2].set_ylabel("Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.savefig(plots_dir / "training_curves_artifacts.png", dpi=200)
mlflow.log_figure(fig, "graficos/training_curves_artifacts.png")
plt.show()
```


```python
def attach_attention_recorders(m: tf.keras.Model) -> None:
    """
    Attach attention layer metadata to a model loaded from disk.

    When a model is loaded via tf.keras.models.load_model(), the custom attributes set during build_multitask_model() (attention_recorders, attention_layer_names) are not persisted in the .keras file. This function restores those attributes by re-mapping the logical task names to their corresponding Keras layer objects in the loaded model.

    Must be called before any attention extraction or interpretability analysis on a model loaded from disk.

    Parameters
    ----------
    m : tf.keras.Model
        Model loaded from disk via tf.keras.models.load_model(). The model must contain layers named 'attention_scores_class_cnn' and 'attention_scores_class_raw'.

    Returns
    -------
    None
        Modifies m in place by setting m.attention_layer_names and m.attention_recorders.
    """
    tap_names = {
        "class_cnn": "attention_scores_class_cnn",
        "class_raw": "attention_scores_class_raw",
    }
    m.attention_layer_names = tap_names
    m.attention_recorders = {
        logical: m.get_layer(layer_name)
        for logical, layer_name in tap_names.items()
        if any(layer.name == layer_name for layer in m.layers)
    }


use_in_memory_model = (
    "model" in globals() and model is not None and "history" in globals()
)

if use_in_memory_model:
    logging.info("Model in memory detected; keeping the trained model in this session")
    # print("Model in memory detected; keeping the trained model in this session.")
    attach_attention_recorders(model)
else:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(
        filepath=model_path,
        custom_objects={"F1Score": F1Score},
    )
    attach_attention_recorders(model)
    logging.info("Model loaded from: %s", model_path)
```

### Test Evaluation

- Keras metrics + custom metrics (accuracy, MAE, RMSE).
- Denormalize energy predictions.



```python
# ─────────────────────────────────────────────────────────────────────────────
# §0.1.16  Test Evaluation — Bayesian Classifier
# · model.evaluate  → métricas Keras (1 pase, MCDropout activo → valor ruidoso
#                     pero comparable con curvas de entrenamiento)
# · mc_predict      → inferencia probabilística (T pases)
# · Temperature Scaling → calibración post-hoc sobre val set
# · Métricas bayesianas → ECE, Brier, NLL, AUROC errores
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Keras evaluate (reemplaza bloque original de 3 outputs) ───────────────
eval_results = model.evaluate(
    x=[X_test, Xg_test],
    y=y_label_test,  # ← solo clasificación, sin ángulo ni energía
    verbose=1,
)
metrics_names = model.metrics_names
eval_dict = {name: float(val) for name, val in zip(metrics_names, eval_results)}
logging.info("Keras evaluate — %s", eval_dict)
# NOTA: la pérdida y accuracy variarán ligeramente en cada ejecución porque
# MCDropout permanece activo. Usar mc_predict para métricas finales reportadas.

# ── 2. MC Dropout inference — val set (nuevo, necesario para Temperature Scaling) ──
logging.info("MC Dropout inference — val set (T=%d)...", MC_T)
mc_val = mc_predict(model, X_val, Xg_val, T=MC_T, batch_size=MC_BATCH)

# ── 3. Temperature Scaling (nuevo) ───────────────────────────────────────────
T_opt, nll_val_min = find_optimal_temperature(mc_val["p_hat"], y_label_val)
logging.info("Temperature Scaling — T* = %.4f  |  NLL_val = %.4f", T_opt, nll_val_min)

# ── 4. MC Dropout inference — test set (reemplaza model.predict) ─────────────
logging.info("MC Dropout inference — test set (T=%d)...", MC_T)
mc_test = mc_predict(model, X_test, Xg_test, T=MC_T, batch_size=MC_BATCH)

p_hat = mc_test["p_hat"]  # p̂_γ     — reemplaza pred_particle
sigma2_ep = mc_test["sigma2_ep"]  # σ²_ep
H = mc_test["H"]  # H[y|x]
MI = mc_test["MI"]  # MI[y,ω|x]
all_preds = mc_test["all_preds"]  # (T, N)

p_hat_cal = temperature_scale(p_hat, T_opt)  # probabilidades calibradas
pred_particle_labels = (p_hat_cal >= 0.5).astype(np.int32)
particle_accuracy = float(np.mean(pred_particle_labels == y_label_test))

# ── 5. pred_summary (reemplaza original — sin columnas de ángulo ni energía) ──
pred_summary = pd.DataFrame(
    {
        "particle_true": y_label_test.astype(int),
        "p_hat": p_hat,  # predicción MC (sin calibrar)
        "p_hat_cal": p_hat_cal,  # predicción calibrada
        "particle_pred_label": pred_particle_labels,
        "sigma2_ep": sigma2_ep,
        "H": H,
        "MI": MI,
        # Covariables físicas — NO son predicciones, se usan para análisis estratificado
        "energy_level_true": y_energy_test,
        "angle_true_deg": y_angle_test.astype(float),
    }
)
pred_summary["particle_correct"] = (
    pred_summary["particle_pred_label"] == pred_summary["particle_true"]
)

# ── 6. Métricas bayesianas (nuevas) ──────────────────────────────────────────

# — 6a. Calibración —
ece_uncal = compute_ece(p_hat, y_label_test)
ece_cal = compute_ece(p_hat_cal, y_label_test)

brier_uncal = compute_brier_score(p_hat, y_label_test)
brier_cal = compute_brier_score(p_hat_cal, y_label_test)

nll_uncal = float(
    -np.mean(
        y_label_test * np.log(p_hat + 1e-8)
        + (1 - y_label_test) * np.log(1 - p_hat + 1e-8)
    )
)
nll_cal = float(
    -np.mean(
        y_label_test * np.log(p_hat_cal + 1e-8)
        + (1 - y_label_test) * np.log(1 - p_hat_cal + 1e-8)
    )
)

calibration_summary = pd.DataFrame(
    {
        "Métrica": ["ECE", "NLL", "Brier Score", "AUC (invariante)"],
        "Sin calibrar": [
            ece_uncal,
            nll_uncal,
            brier_uncal["brier"],
            roc_auc_score(y_label_test, p_hat),
        ],
        "Calibrado": [
            ece_cal,
            nll_cal,
            brier_cal["brier"],
            roc_auc_score(y_label_test, p_hat_cal),
        ],
    }
)
display(calibration_summary.round(4))

# — 6b. Discriminación clásica (sobre predicción calibrada) —
fpr, tpr, _ = roc_curve(y_label_test, p_hat_cal)
roc_auc = auc(fpr, tpr)
gh_metrics = gamma_hadron_metrics(y_label_test, p_hat_cal, threshold=0.5)

particle_report = classification_report(
    y_label_test,
    pred_particle_labels,
    target_names=["Proton", "Photon"],
    output_dict=True,
)
display(pd.DataFrame(particle_report).T.round(4))

# — 6c. AUROC de errores (nueva — valida que σ²_ep predice errores) —
error_auroc = compute_error_auroc(p_hat_cal, y_label_test, sigma2_ep)
logging.info(
    "AUROC de detección de errores: %.4f  (umbral bayesiano: >0.70)",
    error_auroc,
)

additional_metrics = {
    "particle_accuracy_custom": particle_accuracy,
    "ece_uncal": ece_uncal,
    "ece_cal": ece_cal,
    "brier_uncal": brier_uncal["brier"],
    "brier_cal": brier_cal["brier"],
    "nll_uncal": nll_uncal,
    "nll_cal": nll_cal,
    "error_auroc": error_auroc,
    "T_opt": T_opt,
}

```


```python
sns.set_context("poster")
sns.set_style("ticks")

n_bins = 15
bins = np.linspace(0, 1, n_bins + 1)
N = len(y_label_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# ── LEFT: Reliability Diagram ─────────────────────────────────────────────────
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

for p_arr, label_str, color in [
    (p_hat, f"Uncalibrated   ECE={ece_uncal:.4f}", "tab:blue"),
    (
        p_hat_cal,
        f"Calibrated (T*={T_opt:.3f})   ECE={ece_cal:.4f}",
        "tab:orange",
    ),
]:
    xs, ys, errs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask_b = (p_arr >= lo) & (p_arr < hi)
        n_b = mask_b.sum()
        if n_b < 10:
            continue
        p_mean = float(p_arr[mask_b].mean())
        frac = float(y_label_test[mask_b].mean())
        err = float(np.sqrt(frac * (1 - frac) / n_b))
        xs.append(p_mean)
        ys.append(frac)
        errs.append(err)
    xs, ys, errs = np.array(xs), np.array(ys), np.array(errs)
    ax.step(xs, ys, where="mid", color=color, lw=2)
    ax.scatter(xs, ys, color=color, s=60, zorder=3)
    ax.errorbar(xs, ys, yerr=errs, fmt="none", color=color, capsize=4, label=label_str)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Diagram\nCalibration Analysis")
ax.legend(fontsize=12, loc="upper left")
sns.despine(ax=ax)

# ── RIGHT: Calibration Metrics Bar Chart ──────────────────────────────────────
ax2 = axes[1]
metric_names = ["ECE", "NLL", "Brier Score"]
uncal_vals = np.array([ece_uncal, nll_uncal, brier_uncal["brier"]])
cal_vals = np.array([ece_cal, nll_cal, brier_cal["brier"]])

x = np.arange(len(metric_names))
width = 0.35
bars_u = ax2.bar(
    x - width / 2,
    uncal_vals,
    width,
    label="Uncalibrated",
    color="tab:blue",
    alpha=0.85,
)
bars_c = ax2.bar(
    x + width / 2,
    cal_vals,
    width,
    label="Calibrated",
    color="tab:orange",
    alpha=0.85,
)

# Renderizar etiquetas numéricas sobre las barras
for bar in list(bars_u) + list(bars_c):
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h * 1.01,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# --- SOLUCIÓN DE ESPACIADO ---
# 1. Encontrar el valor máximo para calcular el límite dinámico
y_top = max(uncal_vals.max(), cal_vals.max())
# 2. Forzar un límite superior en el eje Y (35% de holgura) para que las etiquetas no choquen arriba
ax2.set_ylim(0, y_top * 1.35)

# Renderizar los porcentajes delta en color verde
for i, (u, c) in enumerate(zip(uncal_vals, cal_vals)):
    delta_pct = (u - c) / u * 100 if u != 0 else 0.0
    ax2.text(
        x[i],
        y_top * 1.10,  # Se posicionan holgadamente sobre las barras
        f"Δ={delta_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="green",
        fontweight="bold",
    )

ax2.set_xticks(x)
ax2.set_xticklabels(metric_names)

# 3. Añadir padding=22 para empujar el título de dos líneas hacia arriba de forma segura
ax2.set_title("Calibration Metrics:\nBefore vs After Temperature Scaling", pad=22)

ax2.legend(fontsize=12)
sns.despine(ax=ax2)

# ── Guardar y Mostrar ─────────────────────────────────────────────────────────
save_path = plots_dir / "calibration_reliability_diagram.png"
fig.savefig(save_path, dpi=200)
mlflow.log_figure(fig, "graficos/calibration_reliability_diagram.png")
logging.info("Guardado: %s", save_path)
plt.show()

```


```python
# ── Bayesian Coverage vs Efficiency (2/2) ────────────────────────────────────
from sklearn.metrics import accuracy_score

sns.set_context("poster")
sns.set_style("ticks")

baseline_acc = accuracy_score(y_label_test, (p_hat_cal >= 0.5).astype(int))

thresholds = np.percentile(sigma2_ep, np.linspace(0, 100, 200))
coverages, accuracies = [], []
for t in thresholds:
    mask = sigma2_ep <= t
    if mask.sum() < 10:
        continue
    coverages.append(float(mask.mean()))
    accuracies.append(
        accuracy_score(y_label_test[mask], (p_hat_cal[mask] >= 0.5).astype(int))
    )
coverages = np.array(coverages)
accuracies = np.array(accuracies)

# ── Métricas adicionales mlflow ───────────────────────────────────────────────
mask_90 = sigma2_ep <= np.percentile(sigma2_ep, 90)
acc_at_90_coverage = accuracy_score(
    y_label_test[mask_90], (p_hat_cal[mask_90] >= 0.5).astype(int)
)
delta_acc_rejection = float(acc_at_90_coverage) - float(baseline_acc)

mlflow.log_metrics(
    {
        "acc_at_90pct_coverage": float(acc_at_90_coverage),
        "delta_acc_rejection": float(delta_acc_rejection),
    }
)
logging.info(
    "acc_at_90pct_coverage=%.4f  delta_acc_rejection=%.4f",
    acc_at_90_coverage,
    delta_acc_rejection,
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# ── LEFT: Accuracy vs Coverage ────────────────────────────────────────────────
ax = axes[0]
ax.plot(coverages, accuracies, color="tab:blue", lw=2)
ax.axhline(
    baseline_acc,
    color="gray",
    linestyle="--",
    lw=1.5,
    label=f"Baseline acc={baseline_acc:.4f}",
)
ax.fill_between(
    coverages,
    accuracies,
    baseline_acc,
    where=(accuracies > baseline_acc),
    alpha=0.15,
    color="tab:green",
    interpolate=True,
)
idx_90 = int(np.argmin(np.abs(coverages - 0.90)))
ax.annotate(
    f"Rejecting top 10% uncertainty\n→ Δacc = {delta_acc_rejection * 100:+.2f}%",
    xy=(coverages[idx_90], accuracies[idx_90]),
    xytext=(max(coverages[idx_90] - 0.30, 0.02), accuracies[idx_90] - 0.02),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=12,
)
ax.set_xlabel("Coverage (fraction of events retained)")
ax.set_ylabel("Accuracy on retained events")
ax.set_title("Bayesian Rejection:\nAccuracy vs Coverage")
ax.legend(fontsize=12)
sns.despine(ax=ax)

# ── CENTER: Distribución de incertidumbre por clase de predicción ─────────────
ax2 = axes[1]
correct = (p_hat_cal >= 0.5).astype(int) == y_label_test.astype(int)
incorrect = ~correct
sns.kdeplot(
    sigma2_ep[correct],
    ax=ax2,
    color="tab:green",
    fill=True,
    alpha=0.35,
    label=f"Correct (n={int(correct.sum())})",
)
sns.kdeplot(
    sigma2_ep[incorrect],
    ax=ax2,
    color="tab:red",
    fill=True,
    alpha=0.35,
    label=f"Incorrect (n={int(incorrect.sum())})",
)
ax2.set_xlabel(r"Epistemic variance $\sigma^2_{ep}$")
ax2.set_title("Uncertainty Distribution by\nPrediction Correctness")
ax2.legend(fontsize=12)
sns.despine(ax=ax2)

# ── RIGHT: Accuracy por cuantil de sigma2_ep ─────────────────────────────────
ax3 = axes[2]
n_q = 10
q_edges = np.percentile(sigma2_ep, np.linspace(0, 100, n_q + 1))
bin_centers_q, bin_accs_q = [], []
for lo, hi in zip(q_edges[:-1], q_edges[1:]):
    mask_q = (sigma2_ep >= lo) & (sigma2_ep <= hi)
    if mask_q.sum() < 10:
        continue
    bin_centers_q.append(float((lo + hi) / 2))
    bin_accs_q.append(
        accuracy_score(y_label_test[mask_q], (p_hat_cal[mask_q] >= 0.5).astype(int))
    )
bin_labels_q = [f"{c:.2e}" for c in bin_centers_q]
ax3.bar(range(len(bin_accs_q)), bin_accs_q, color="tab:blue", alpha=0.8)
ax3.axhline(
    baseline_acc,
    color="gray",
    linestyle="--",
    lw=1.5,
    label=f"Baseline acc={baseline_acc:.4f}",
)
ax3.set_xticks(range(len(bin_accs_q)))
ax3.set_xticklabels(bin_labels_q, rotation=45, ha="right", fontsize=10)
ax3.set_xlabel(r"$\sigma^2_{ep}$ bin (quantile)")
ax3.set_ylabel("Accuracy")
ax3.set_title("Accuracy by Uncertainty Quantile")
ax3.legend(fontsize=12)
sns.despine(ax=ax3)

save_path = plots_dir / "bayesian_coverage_efficiency.png"
fig.savefig(save_path, dpi=200)
mlflow.log_figure(fig, "graficos/bayesian_coverage_efficiency.png")
logging.info("Guardado: %s", save_path)
plt.show()
```


```python
# ── 7. Artifacts (reemplaza bloque original) ──────────────────────────────────

# — 7a. Attention layer mapping (actualizado — 2 tareas en vez de 6) —
attention_map = {
    "class_cnn": "attention_scores_class_cnn",
    "class_raw": "attention_scores_class_raw",
}
attention_json = model_dir / "attention_layers.json"
with attention_json.open("w", encoding="utf-8") as fh:
    json.dump(attention_map, fh, indent=2)
logging.info("Attention layer mapping saved to: %s", attention_json)

history_csv = model_dir / "training_history.csv"  # definición de ruta, siempre
SAVE_ARTIFACTS = True  # Controla si se guardan los artifacts generados en esta␣sesión

# — 7b. Training history —
if SAVE_ARTIFACTS:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv, index=False)
    logging.info("Training history saved to: %s", history_csv)

# — 7c. Metrics JSON (reemplaza original — sin ángulo ni energía) —
metrics_path = model_dir / "evaluation_metrics.json"
with metrics_path.open("w", encoding="utf-8") as fh:
    json.dump(
        {
            "keras_metrics": eval_dict,
            "custom_metrics": additional_metrics,
            "calibration": {
                "T_opt": T_opt,
                "ece_uncal": ece_uncal,
                "ece_cal": ece_cal,
            },
        },
        fh,
        indent=2,
    )

# — 7d. Predictions NPZ (reemplaza original — arrays bayesianos) —
pred_npz_path = ARTIFACTS_DIR / "test_predictions.npz"
np.savez_compressed(
    pred_npz_path,
    # Predicciones
    p_hat=p_hat.astype(np.float32),
    p_hat_cal=p_hat_cal.astype(np.float32),
    particle_label_pred=pred_particle_labels,
    particle_label_true=y_label_test,
    # Incertidumbre
    sigma2_ep=sigma2_ep.astype(np.float32),
    H=H.astype(np.float32),
    MI=MI.astype(np.float32),
    all_preds=all_preds.astype(np.float32),  # (T, N) — para SHAP, LRP
    # Covariables físicas (para análisis estratificado, no son predicciones)
    energy_level_true=np.array(y_energy_test, dtype="<U8"),
    angle_true_deg=y_angle_test.astype(np.float32),
    # Calibración
    T_opt=np.array([T_opt], dtype=np.float32),
)
logging.info("Test predictions saved to: %s", pred_npz_path)

```


```python
# — 7e. Metadata —
metadata["artifacts"] = {
    "history_csv": str(history_csv),
    "model_path": str(model_path),
    "metrics_path": str(metrics_path),
    "predictions_npz": str(pred_npz_path),
    "detector_catalog_csv": str(ARTIFACTS_DIR / "detector_catalog.csv"),
    "attention_json": str(attention_json),
}
metadata["keras_metrics"] = eval_dict
metadata["custom_metrics"] = additional_metrics
with metadata_path.open("w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2)

# ── 8. MLflow (reemplaza bloque original — métricas actualizadas) ─────────────
mlflow.log_params({"temperature_T_opt": round(T_opt, 4), "MC_T": MC_T})

mlflow.log_metrics(
    {
        # Discriminación
        "auc": float(roc_auc),
        "q_factor": float(gh_metrics["q_factor"]),
        "gamma_efficiency": float(gh_metrics["efficiency_gamma"]),
        "hadron_contamination": float(gh_metrics["contamination_hadron"]),
        "f1_score": float(eval_dict.get("f1_score", float("nan"))),
        # Calibración
        "ece_uncal": float(ece_uncal),
        "ece_cal": float(ece_cal),
        "brier_uncal": float(brier_uncal["brier"]),
        "brier_cal": float(brier_cal["brier"]),
        "nll_uncal": float(nll_uncal),
        "nll_cal": float(nll_cal),
        # Validez bayesiana
        "error_auroc": float(error_auroc),
    }
)

mlflow.log_artifact(str(model_path))
mlflow.log_artifact(str(metrics_path))

# Firma del modelo (1 output en vez de 3)
sample_input = {"seq_input": X_val[:5], "global_input": Xg_val[:5]}
sample_output = model.predict([X_val[:5], Xg_val[:5]])
signature = infer_signature(
    sample_input,
    {"particle_output": sample_output},
)
mlflow.tensorflow.log_model(
    model,
    artifact_path="condor_bayesian_classifier",
    signature=signature,
    registered_model_name="Modelo_Condor_Bayesiano",
)
logging.info("Metadata (post-evaluation) saved to: %s", metadata_path)
logging.info("Metadata post training and evaluation saved to: %s", metadata_path)
```

### Gamma/Hadron Results

- Confusion matrix, classification report.
- ROC/AUC, score histograms.
- Q-factor, efficiencies, Li-Ma significance.



```python
cm_particle = confusion_matrix(
    pred_summary["particle_true"], pred_summary["particle_pred_label"]
)
particle_report = classification_report(
    pred_summary["particle_true"],
    pred_summary["particle_pred_label"],
    target_names=["Proton", "Photon"],
    output_dict=True,
)
display(pd.DataFrame(particle_report).T.round(4))

fpr, tpr, _ = roc_curve(pred_summary["particle_true"], pred_summary["p_hat_cal"])
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

sns.heatmap(cm_particle, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_xlabel(r"$\hat{p}$")
axes[0].set_ylabel(r"$p_{\text{true}}$")
axes[0].set_title("Confusion Matrix \n Gamma-Hadron Discrimination")
axes[0].set_xticklabels(["Proton", r"$\gamma$"])
axes[0].set_yticklabels(["Proton", r"$\gamma$"])

sns.histplot(
    data=pred_summary,
    x="p_hat_cal",
    hue="particle_true",
    bins=30,
    element="step",
    common_norm=False,
    palette={0: "tab:blue", 1: "tab:orange"},
    ax=axes[1],
)
axes[1].set_title(r"P($\gamma$ | Proton) Distribution")
axes[1].set_xlabel(r"P($\gamma$)")
axes[1].legend(labels=[r"$\gamma$", "Proton"])

axes[2].plot(fpr, tpr, color="tab:green", label=f"AUC = {roc_auc:.3f}")
axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve")
axes[2].legend(loc="lower right")

fig.savefig(plots_dir / "particle_predictions_overview.png", dpi=200)
mlflow.log_figure(fig, "graficos/particle_predictions_overview.png")
plt.show()
```


```python
gh_metrics = gamma_hadron_metrics(
    pred_summary["particle_true"].to_numpy(), pred_summary["p_hat_cal"].to_numpy()
)

```



### Zenith Angle Reconstruction Results

- Error distributions (abs/rel).
- Profile plots: predicted vs true.
- Metrics: MAE, RMSE, bias, PSF68.



```python
# ----------------------------------------- #
# Classification performance by zenith angle  #
# ----------------------------------------- #
theta_true = pred_summary["angle_true_deg"]
particle_series = pred_summary["particle_true"]

particle_names = {0: "Proton", 1: r"$\gamma$"}
particle_palette = {0: "tab:blue", 1: "tab:orange"}
angle_bin_edges = [0, 10, 20, 30, 40]
angle_bin_labels = ["0-10", "10-20", "20-30", "30-40"]

rows = []
for lo, hi, lbl in zip(angle_bin_edges[:-1], angle_bin_edges[1:], angle_bin_labels):
    mask = (theta_true >= lo) & (theta_true < hi)
    n = int(mask.sum())
    if n < 10:
        continue
    y_t = pred_summary.loc[mask, "particle_true"]
    y_p = pred_summary.loc[mask, "p_hat_cal"]
    y_l = pred_summary.loc[mask, "particle_pred_label"]
    acc = float((y_t == y_l).mean())
    fpr_b, tpr_b, _ = roc_curve(y_t, y_p)
    auc_b = float(auc(fpr_b, tpr_b))
    mean_H = float(pred_summary.loc[mask, "H"].mean())
    rows.append({"bin": lbl, "n": n, "acc": acc, "auc": auc_b, "mean_H": mean_H})

angle_perf = pd.DataFrame(rows)
display(angle_perf.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# --- AXES 0: Accuracy by angle bin ---
axes[0].bar(angle_perf["bin"], angle_perf["acc"], color="steelblue", edgecolor="k")
axes[0].set_ylim(0.5, 1.0)
axes[0].set_xlabel(r"Zenith angle bin ($^\circ$)")
axes[0].set_ylabel("Accuracy")
axes[0].set_title(r"Accuracy vs. $\theta_{\mathrm{true}}$ bin")
for bar, val in zip(axes[0].patches, angle_perf["acc"]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# --- AXES 1: AUC by angle bin ---
axes[1].bar(angle_perf["bin"], angle_perf["auc"], color="darkorange", edgecolor="k")
axes[1].set_ylim(0.5, 1.0)
axes[1].set_xlabel(r"Zenith angle bin ($^\circ$)")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title(r"ROC-AUC vs. $\theta_{\mathrm{true}}$ bin")
for bar, val in zip(axes[1].patches, angle_perf["auc"]):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# --- AXES 2: p_hat_cal distribution by angle bin ---
pred_summary["angle_bin"] = pd.cut(
    theta_true, bins=angle_bin_edges, labels=angle_bin_labels, right=False
)
bin_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for lbl, color in zip(angle_bin_labels, bin_colors):
    subset = pred_summary[pred_summary["angle_bin"] == lbl]["p_hat_cal"]
    if len(subset) > 0:
        sns.kdeplot(subset, ax=axes[2], label=lbl, color=color, fill=True, alpha=0.3)
axes[2].set_xlabel(r"$\hat{p}_{\gamma}^{\mathrm{cal}}$")
axes[2].set_ylabel("Density")
axes[2].set_title(r"$\hat{p}_{\gamma}^{\mathrm{cal}}$ distribution by zenith angle bin")
axes[2].legend(title=r"$\theta_{\mathrm{true}}$ ($^\circ$)")

fig.suptitle("Gamma-Hadron Classification Performance by Zenith Angle", fontsize=13)
fig.savefig(plots_dir / "angle_stratified_classification.png", dpi=200)
mlflow.log_figure(fig, "graficos/angle_stratified_classification.png")
plt.show()

```

### Stratified ROC Analyses

- ROC by energy bin (3E2, 5E2, 8E2).
- ROC by zenith bins (0–10, 10–20, 20–30, 30–40°).



```python
# ============================================================
# ROC CURVES: STRATIFIED BY PHYSICS PARAMETERS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC by energy level
ax1 = axes[0]
for energy_level in ["3E2", "5E2", "8E2"]:
    mask = pred_summary["energy_level_true"] == energy_level
    if mask.sum() > 10:
        fpr_e, tpr_e, _ = roc_curve(
            pred_summary[mask]["particle_true"],
            pred_summary[mask]["p_hat_cal"],
        )
        auc_e = auc(fpr_e, tpr_e)
        ax1.plot(fpr_e, tpr_e, label=f"{energy_level} (AUC={auc_e:.3f})", linewidth=2)

ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax1.set_xlabel("False Positive Rate (Hadron Contamination)")
ax1.set_ylabel("True Positive Rate (Gamma Efficiency)", fontsize=16)
ax1.set_title("ROC Curves by Energy Level")
ax1.legend()
ax1.grid(alpha=0.3)

# ROC by zenith angle
ax2 = axes[1]
angle_bins = [(0, 10), (10, 20), (20, 30), (30, 40)]
for angle_min, angle_max in angle_bins:
    mask = (pred_summary["angle_true_deg"] >= angle_min) & (
        pred_summary["angle_true_deg"] < angle_max
    )
    if mask.sum() > 10:
        fpr_a, tpr_a, _ = roc_curve(
            pred_summary[mask]["particle_true"],
            pred_summary[mask]["p_hat_cal"],
        )
        auc_a = auc(fpr_a, tpr_a)
        ax2.plot(
            fpr_a,
            tpr_a,
            label=f"{angle_min}-{angle_max}° (AUC={auc_a:.3f})",
            linewidth=2,
        )

ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves by Zenith Angle")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "roc_stratified.png", dpi=200)
mlflow.log_figure(fig, "graficos/roc_stratified.png")
plt.show()
```

### Migration Matrices

- Energy migration (true vs reconstructed).
- Angle migration.
- Joint energy–particle migration.



```python
# ============================================================
# CONFUSION MATRIX POR BIN DE ENERGÍA
# ============================================================

energy_levels = ["3E2", "5E2", "8E2"]
fig, axes = plt.subplots(
    1, len(energy_levels), figsize=(18, 5), constrained_layout=True
)

for ax, energy_level in zip(axes, energy_levels):
    mask = pred_summary["energy_level_true"] == energy_level
    subset = pred_summary[mask]
    if len(subset) < 2:
        ax.set_title(f"Energy: {energy_level}\nInsuficientes datos")
        continue
    cm = confusion_matrix(subset["particle_true"], subset["particle_pred_label"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["Proton", r"$\gamma$"],
        yticklabels=["Proton", r"$\gamma$"],
    )
    ax.set_title(f"Energy bin: {energy_level}\n(n={len(subset)})")
    ax.set_xlabel(r"$\hat{p}$ (predicted)")
    ax.set_ylabel(r"$p_{\text{true}}$")

plt.savefig(
    plots_dir / "particle_confusion_by_energy.png", dpi=200, bbox_inches="tight"
)
mlflow.log_figure(fig, "graficos/particle_confusion_by_energy.png")
plt.show()

```

### Consolidated Performance

- Summary table of primary metrics per task (AUC, PSF68, energy bias/resolution).
- Save CSV.



```python
# ============================================================
# CONSOLIDATED PERFORMANCE TABLE — BAYESIAN CLASSIFIER
# ============================================================

summary_table = pd.DataFrame(
    {
        "Metric": [
            "AUC-ROC",
            "Q-factor",
            "Gamma Efficiency",
            "Hadron Contamination",
            "ECE (calibrated)",
            "Brier Score (calibrated)",
            "Error AUROC",
            "T_opt",
        ],
        "Value": [
            f"{roc_auc:.4f}",
            f"{gh_metrics['q_factor']:.3f}",
            f"{gh_metrics['efficiency_gamma']:.3f}",
            f"{gh_metrics['contamination_hadron']:.3f}",
            f"{ece_cal:.4f}",
            f"{brier_cal['brier']:.4f}",
            f"{error_auroc:.4f}",
            f"{T_opt:.4f}",
        ],
    }
)

display(summary_table)
summary_table.to_csv(ARTIFACTS_DIR / "performance_summary.csv", index=False)

```


```python
mlflow.end_run()
logging.info(
    "MLflow run completed — experiment: %s, run: %s", MLFLOW_EXPERIMENT_NAME, run_name
)
```

### Feature Ablation

- Zero/mean/median replacement per feature.
- Impact on particle accuracy, angle MAE, energy MAE.
- Bar plots (absolute and relative drops).
- Summary CSVs.



```python
# ============================================================
# FEATURE ABLATION STUDY
# ============================================================
"""
Objective: Measure the importance of each feature by REMOVING it from the model
Difference vs PFI: 
   - PFI: permutes values (maintains distribution)
   - Ablation: replaces with neutral value (mean/median/zero)
"""


# ----------------------------- #
# Ablation Functions            #
# ----------------------------- #
def ablate_feature(
    X_seq: np.ndarray,
    feature_idx: int,
    strategy: str = "mean",
) -> np.ndarray:
    """
    Replace a sequence feature with a neutral value across all events.

    Ablation replaces a feature with a constant neutral value to measure its contribution to model performance. Unlike permutation feature importance (PFI), ablation removes the feature's information entirely rather than shuffling it, making it sensitive to both the feature's marginal and interaction effects.

    Parameters
    ----------
    X_seq : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features).
    feature_idx : int
        Index of the feature to ablate along the last axis.
        See FEATURE_NAMES for the mapping from index to feature name.
    strategy : str, optional
        Replacement strategy:
        - 'zero'  : Replace with 0.0. Appropriate for features where zero is a physically meaningful absence of signal.
        - 'mean'  : (Default value) replace with the feature mean across all events and time steps, ignoring NaN values.
        - 'median': Replace with the feature median, more robust to outliers than the mean.
        Default is 'mean'.

    Returns
    -------
    np.ndarray
        Copy of X_seq with the selected feature replaced by the neutral value. The original array is not modified.

    Raises
    ------
    ValueError
        If strategy is not one of 'zero', 'mean', or 'median'.
    """
    X_ablated = np.array(X_seq, copy=True)
    feature_data = X_seq[:, :, feature_idx]

    # Apply ablation strategy
    if strategy == "zero":
        replacement_val = 0.0
    elif strategy == "mean":
        replacement_val = np.nanmean(feature_data)
    elif strategy == "median":
        replacement_val = np.nanmedian(feature_data)
    else:
        raise ValueError(f"Estrategia de ablación no soportada: '{strategy}'")
    X_ablated[:, :, feature_idx] = replacement_val

    return X_ablated

```


```python
def compute_task_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
) -> float:
    """
    Compute the primary evaluation metric for a given task.

    Returns accuracy for binary classification (particle) and mean absolute error for regression tasks (angle, energy). Used consistently across baseline and permuted predictions in the ablation and PFI studies to ensure comparability of importance scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values for the task.
    y_pred : np.ndarray
        Model predictions. For 'particle', these are raw probabilities in [0, 1] that are thresholded at 0.5 internally.
    task_name : str
        Task identifier. Must be one of:
        - 'particle' : binary classification → accuracy.
        - 'angle'    : zenith angle regression → MAE in degrees.
        - 'energy'   : energy regression → MAE in GeV.

    Returns
    -------
    float
        Task-specific scalar metric. Higher is better for 'particle' (accuracy); lower is better for 'angle' and 'energy' (MAE).
    """
    return float(accuracy_score(y_true.astype(int), (y_pred >= 0.5).astype(int)))

```


```python
def run_ablation_study(
    model: tf.keras.Model,
    x_seq: np.ndarray,
    x_global: np.ndarray,
    y_true_dict: dict[str, np.ndarray],
    feature_names: list[str],
    strategy: str = "mean",
    batch_size: int = 256,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs a complete ablation study on the given model.

    Args:
        model: Trained Keras model.
        x_seq: Input sequences (batch, seq_len, n_features).
        x_global: Global features (batch, n_global_features).
        y_true_dict: Dict with ground truth per task {'particle': array}.
        feature_names: List of dataset feature names that are being ablated.
        strategy: Imputation strategy ('zero', 'mean', or 'median').
        batch_size: Batch size for inference.
        verbose: Show progress and prints.

    Returns:
        Tuple containing:
            - results_df: DataFrame with ablation results.
            - baseline_scores: Dictionary with the baseline metrics.
    """
    if verbose:
        logging.info("📊 [Ablation study] Calculating baseline (without ablation)...")

    # 1. -----------------    Baseline prediction    -----------------
    baseline_preds = mc_predict(model, x_seq, x_global, T=30, batch_size=batch_size)
    pred_particle_base = baseline_preds["p_hat"]

    # Calculate baseline scores
    baseline_scores = {
        "particle": compute_task_metric(
            y_true_dict["particle"], pred_particle_base, "particle"
        ),
    }

    if verbose:
        logging.info("✅ Baseline Scores:")
        for task, score in baseline_scores.items():
            logging.info("\t%s: %.4f", task, score)

    # 2. -----------------    Ablation for each feature    -----------------
    results: list[dict[str, Any]] = []

    iterator = enumerate(feature_names)
    if verbose:
        iterator = tqdm(
            list(iterator), desc="Ablating features"
        )  # Barra de progreso en la ejecución del bucle, para monitoreo.

    for feat_idx, feat_name in iterator:
        # Ablate feature
        x_ablated = ablate_feature(x_seq, feat_idx, strategy=strategy)

        # Predict with ablated model
        ablated_preds = mc_predict(
            model, x_ablated, x_global, T=30, batch_size=batch_size
        )
        pred_particle_abl = ablated_preds["p_hat"]

        # Calculate scores
        ablated_scores = {
            "particle": compute_task_metric(
                y_true_dict["particle"], pred_particle_abl, "particle"
            ),
        }

        # Calculate degradation (accuracy: baseline - ablated)
        for task in ["particle"]:
            baseline = baseline_scores[task]
            ablated = ablated_scores[task]
            importance = baseline - ablated

            results.append(
                {
                    "feature": feat_name,
                    "task": task,
                    "strategy": strategy,
                    "baseline_score": baseline,
                    "ablated_score": ablated,
                    "importance": importance,
                    "relative_importance": (
                        (importance / baseline) * 100 if baseline != 0 else 0.0
                    ),
                }
            )

    results_df = pd.DataFrame(results)

    return results_df, baseline_scores

```


```python
# ----------------------------- #
# Execute Ablation Study       #
# ----------------------------- #

lines = [
    "",
    "=" * 80,
    "FEATURE ABLATION STUDY",
    "=" * 80,
    f"\nEstrategia de ablation: MEAN (reemplazar con promedio)",
    f"Dataset: Test set ({len(X_test)} samples)",
    f"Features analizadas: {FEATURE_NAMES}",
]
logging.info("\n".join(lines))

y_true_dict = {
    "particle": y_label_test,
}

ABLATION_STRATEGY_USED = "median"  # Se puede cambiar a "zero" o "mean" según se quiera probar diferentes estrategias de ablación.
# Execute ablation
ablation_results, baseline_scores = run_ablation_study(
    model=model,
    x_seq=X_test,
    x_global=Xg_test,
    feature_names=FEATURE_NAMES,
    y_true_dict=y_true_dict,
    strategy=ABLATION_STRATEGY_USED,
    batch_size=64,
    verbose=True,
)
```


```python
# / ── Metadata post ablación ───────────────────────────────────────────────────
metadata["ablation_config"] = {
    "strategy_used": ABLATION_STRATEGY_USED,
    "features_ablated": FEATURE_NAMES,
    "n_test_samples": len(X_test),
}

with metadata_path.open("w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2)
logging.info("Metadata (post-ablation) saved to: %s", metadata_path)

logging.info("Metadata post ablation saved to: %s", metadata_path)

```


```python
# ============================================================
# RESULTS VISUALIZATION — FEATURE ABLATION STUDY (XAI)
# ============================================================

# Cambiamos la cuadrícula a 1 fila y 2 columnas (Lado a lado)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Definimos variables fijas para la única tarea actual
task_title = "Gamma-Hadron Discrimination"
metric_name = "Accuracy"  # O "AUC" si cambiaste la métrica de evaluación en la ablación

# 1. Filtrar y ordenar los resultados exclusivamente para la tarea 'particle'
subset = ablation_results[ablation_results["task"] == "particle"].copy()
subset = subset.sort_values("importance", ascending=False)

# Generar paleta de colores basada en el número de variables analizadas
colors = sns.color_palette("rocket", len(subset))

# ------------------------------------------------------------
# PLOT 1: Absolute Importance (Izquierda)
# ------------------------------------------------------------
ax1 = axes[0]
bars1 = ax1.barh(
    subset["feature"],
    subset["importance"],
    color=colors,
    edgecolor="black",
    linewidth=1.5,
)
ax1.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
ax1.set_xlabel(f"Δ {metric_name}\n(Higher = More Important)", fontsize=11)
ax1.set_title(
    f"{task_title}\nAbsolute Feature Importance", fontsize=12, fontweight="bold"
)
ax1.grid(axis="x", alpha=0.3)

# Añadir los valores numéricos al lado de cada barra
for bar, val in zip(bars1, subset["importance"]):
    width = bar.get_width()
    label_x = width + (
        0.01 * ax1.get_xlim()[1] if width > 0 else -0.01 * ax1.get_xlim()[1]
    )
    ax1.text(
        label_x,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        ha="left" if width > 0 else "right",
        va="center",
        fontsize=9,
    )

# ------------------------------------------------------------
# PLOT 2: Relative Importance (Derecha)
# ------------------------------------------------------------
ax2 = axes[1]
bars2 = ax2.barh(
    subset["feature"],
    subset["relative_importance"],
    color=colors,
    edgecolor="black",
    linewidth=1.5,
)
ax2.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
ax2.set_xlabel("Relative Importance (%)", fontsize=11)
ax2.set_title(
    f"{task_title}\nRelative Impact on {metric_name}", fontsize=12, fontweight="bold"
)
ax2.grid(axis="x", alpha=0.3)

# Añadir los valores porcentuales al lado de cada barra
for bar, val in zip(bars2, subset["relative_importance"]):
    width = bar.get_width()
    label_x = width + (0.5 if width > 0 else -0.5)
    ax2.text(
        label_x,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        ha="left" if width > 0 else "right",
        va="center",
        fontsize=9,
    )

# Ajustes finales y guardado de artefactos
plt.tight_layout()
plt.savefig(plots_dir / "feature_ablation_study.png", dpi=200, bbox_inches="tight")
mlflow.log_figure(fig, "graficos/feature_ablation_study.png")
plt.show()

```


```python
# ============================================================
# 📋 FEATURE ABLATION SUMMARY (SINGLE-TASK)
# ============================================================

summary_lines = [
    "",
    "=" * 80,
    "📋 FEATURE ABLATION SUMMARY",
    "=" * 80,
]
logging.info("\n".join(summary_lines))
# ------------------------------------------------------------
# Pivot single-task | Mantiene la tabla feature -> importance
# ------------------------------------------------------------
pivot_table = ablation_results.pivot_table(
    index="feature",
    columns="task",
    values="importance",
    aggfunc="mean",
).round(4)

# Extraer solamente la tarea particle (Programación defensiva)
pivot_table = pivot_table[["particle"]]

# ------------------------------------------------------------
# Ranking de importancia (Método Denso)|Mayor importance = mayor impacto sobre accuracy
# ------------------------------------------------------------

pivot_table["rank_particle"] = pivot_table["particle"].rank(
    ascending=False,
    method="dense",
)
# Ordenar features por impacto
pivot_table = pivot_table.sort_values(
    "particle",
    ascending=False,
)

display(pivot_table)
# Guardar resultados localmente
ablation_results.to_csv(
    ARTIFACTS_DIR / "diagnostics" / "feature_ablation_results.csv",
    index=False,
)

pivot_table.to_csv(ARTIFACTS_DIR / "diagnostics" / "feature_ablation_summary.csv")
summary_lines = [
    "",
    "Results saved to:",
    f"   - {ARTIFACTS_DIR / 'diagnostics' / 'feature_ablation_results.csv'}",
    f"   - {ARTIFACTS_DIR / 'diagnostics' / 'feature_ablation_summary.csv'}",
]

logging.info("\n".join(summary_lines))
# Registro completo en MLflow
mlflow.log_artifact(str(ARTIFACTS_DIR / "diagnostics" / "feature_ablation_results.csv"))
mlflow.log_artifact(str(ARTIFACTS_DIR / "diagnostics" / "feature_ablation_summary.csv"))
```


```python
# ----------------------------- #
# Physical Insights             #
# ----------------------------- #

tasks = ["particle"]
task_titles = ["Gamma-Hadron Discrimination"]

insights_header = ["", "=" * 80, "PHYSICAL INSIGHTS FROM ABLATION", "=" * 80]
logging.info("\n".join(insights_header))

# Top 3 features per task
for task, title in zip(tasks, task_titles):
    top3 = ablation_results[ablation_results["task"] == task].nlargest(3, "importance")

    task_lines = [f"\n{title}:"]
    for i, row in enumerate(top3.itertuples(), 1):
        task_lines.append(
            f"   {i}. {row.feature:15s}: Importance = {row.importance:.4f} ({row.relative_importance:+.1f}%)"
        )

    logging.info("\n".join(task_lines))

# Feature importance summary
logging.info("\n\nFEATURE IMPORTANCE SUMMARY:")

feat_summary = (
    ablation_results.groupby("feature")["importance"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)
feat_summary = feat_summary.sort_values("mean", ascending=False)
display(feat_summary)

logging.info("\n" + "=" * 80)

```

### Permutation Feature Importance

- Per-sequence feature PFI across all heads.
- Triptych bar plots per task.
- Save CSV and figures.



```python
# Permutation Feature Importance only for Sequence Features

OUTPUT_SPECS = {
    "particle": {
        "metric_name": "accuracy",
        "metric": lambda pred: accuracy_score(
            y_label_test.astype(int), (pred >= 0.5).astype(int)
        ),
        "higher_is_better": True,
    },
}


def _predict_all_heads(
    seq_input: np.ndarray,
    global_input: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """
    Run inference on the particle classification head.

    Uses a single MC Dropout forward pass (T=1) for PFI purposes.
    Statistical stability is achieved via n_repeats in the outer loop
    rather than MC averaging, reducing total forward passes by ~30x
    compared to T=30.

    Parameters
    ----------
    seq_input : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features).
    global_input : np.ndarray
        Global feature array of shape (n_events, n_global_features).
    batch_size : int, optional
        Number of events per inference batch. Default is 256.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with key:
        - 'particle': predicted gamma probabilities, shape (n_events,).
    """
    result = mc_predict(model, seq_input, global_input, T=1, batch_size=batch_size)
    return {"particle": result["p_hat"]}


def _permute_sequence_feature(
    seq_data: np.ndarray,
    feat_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute one feature across all events independently per time step.

    Implements column-wise permutation for PFI using a fully vectorized
    numpy operation, replacing the previous Python loop over timesteps.
    For each time step, feature values are shuffled across the batch
    dimension by generating a random sort index matrix and applying it
    in a single array indexing operation.

    Parameters
    ----------
    seq_data : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features).
        The original array is not modified.
    feat_idx : int
        Index of the feature to permute along the last axis.
        See FEATURE_NAMES for the mapping from index to feature name.
    rng : np.random.Generator
        NumPy random generator instance. A fresh generator derived from
        rng_master should be passed for each repeat to ensure
        independence across permutation repeats.

    Returns
    -------
    np.ndarray
        Copy of seq_data with feature feat_idx permuted independently
        at each time step. Shape is identical to seq_data.
    """
    permuted = seq_data.copy()
    feature_view = permuted[:, :, feat_idx]  # (n_events, seq_len)
    shuffle_idx = rng.random(feature_view.shape).argsort(axis=0)
    permuted[:, :, feat_idx] = feature_view[
        shuffle_idx, np.arange(feature_view.shape[1])
    ]
    return permuted


def compute_pfi_sequence_features(
    model: tf.keras.Model,
    X_seq: np.ndarray,
    X_global: np.ndarray,
    *,
    n_repeats: int = 15,
    random_state: int = 42,
    batch_size: int = 256,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute Permutation Feature Importance (PFI) for all sequence features.

    For each feature in FEATURE_NAMES, permutes its values across the
    batch independently at each time step and measures the resulting
    degradation in the particle classification accuracy relative to
    the baseline. The experiment is repeated n_repeats times with
    independent random permutations to estimate the variance of the
    importance scores.

    Importance is defined as baseline − permuted_mean for accuracy-based
    metrics: positive values indicate the feature contributes positively
    to classification performance.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Bayesian classifier (single-task, particle only).
    X_seq : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features).
    X_global : np.ndarray
        Global feature array of shape (n_events, n_global_features).
    n_repeats : int, keyword-only
        Number of independent permutation repeats per feature.
        Default is 15. Higher values reduce variance of importance
        estimates. With T=1 in _predict_all_heads, n_repeats=15
        provides equivalent statistical stability to the previous
        n_repeats=8 with T=30 at ~30x lower compute cost.
    random_state : int, keyword-only
        Master random seed for reproducibility. Default is 42.
    batch_size : int, keyword-only
        Inference batch size. Default is 256.

    Returns
    -------
    tuple[dict[str, float], pd.DataFrame]
        - baseline_scores : dict with key 'particle' and its baseline
          accuracy on the unperturbed input.
        - results : DataFrame with one row per feature, sorted by
          importance_delta descending. Columns: feature, task,
          metric_name, baseline, permuted_mean, permuted_std,
          importance_delta.
    """
    rng_master = np.random.default_rng(random_state)
    baseline_preds = _predict_all_heads(X_seq, X_global, batch_size=batch_size)
    baseline_scores = {
        task: spec["metric"](baseline_preds[task])
        for task, spec in OUTPUT_SPECS.items()
    }

    records = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        task_scores = {task: [] for task in OUTPUT_SPECS}
        for _ in range(n_repeats):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            X_perm = _permute_sequence_feature(X_seq, feat_idx, rng)
            perm_preds = _predict_all_heads(X_perm, X_global, batch_size=batch_size)
            for task, spec in OUTPUT_SPECS.items():
                task_scores[task].append(spec["metric"](perm_preds[task]))

        for task, scores in task_scores.items():
            scores = np.asarray(scores, dtype=float)
            baseline = baseline_scores[task]
            mean_score = float(scores.mean())
            std_score = float(scores.std(ddof=1))
            delta = (
                baseline - mean_score
                if OUTPUT_SPECS[task]["higher_is_better"]
                else mean_score - baseline
            )
            records.append(
                {
                    "feature": f"sequence::{feat_name}",
                    "task": task,
                    "metric_name": OUTPUT_SPECS[task]["metric_name"],
                    "baseline": baseline,
                    "permuted_mean": mean_score,
                    "permuted_std": std_score,
                    "importance_delta": delta,
                }
            )

    results = (
        pd.DataFrame(records)
        .sort_values(["task", "importance_delta"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return baseline_scores, results


baseline_scores, pfi_results = compute_pfi_sequence_features(
    model,
    X_test,
    Xg_test,
    n_repeats=15,
    random_state=42,
    batch_size=256,
)

baseline_lines = ["Baseline metrics by head:"]
for task, value in baseline_scores.items():
    baseline_lines.append(f"  {task}: {value:.4f}")

logging.info("\n".join(baseline_lines))

display(pfi_results)
pfi_results.to_csv(
    ARTIFACTS_DIR / "diagnostics" / "pfi_all_heads_sequence_only.csv", index=False
)

```


```python
def _friendly_feature_name(raw_name: str) -> str:
    """
    Extract the human-readable feature name from a namespaced PFI identifier.

    PFI results store feature names with a 'sequence::' namespace prefix (e.g. 'sequence::particle_count') to allow future extension with global feature importance. This function strips the prefix for display in plots and tables.

    Parameters
    ----------
    raw_name : str
        Namespaced feature name, e.g. 'sequence::particle_count'.

    Returns
    -------
    str
        Feature name without namespace prefix, e.g. 'particle_count'.
        If no '::' separator is present, the original string is returned.
    """
    return raw_name.split("::", 1)[-1]


def plot_pfi_triptych(
    pfi_results: pd.DataFrame,
    top_n: int | None = None,
    output_path: Path | None = None,
):
    """
    Plot a three-panel bar chart of Permutation Feature Importance by task.

    Each panel shows the importance_delta for all sequence features for one task (particle classification, angle reconstruction, energy estimation), sorted in descending order of importance. Bar values are annotated directly on the chart.

    Parameters
    ----------
    pfi_results : pd.DataFrame
        Output of compute_pfi_sequence_features(). Must contain columns: task, feature, importance_delta.
    top_n : int, optional
        If provided, only the top_n most important features are shown per panel. If None, all features are shown. Default is None.
    output_path : Path, optional
        If provided, the figure is saved to this path at 200 dpi before displaying. If None, the figure is only displayed. Default is None.

    """
    tasks_order = [
        ("particle", "Gamma-Hadron Discrimination PFI", "Decrease in accuracy"),
    ]
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axes = [axes]
    sns.set_style("whitegrid")

    for ax, (task, title, ylabel) in zip(axes, tasks_order):
        subset = pfi_results[pfi_results["task"] == task].copy()
        subset = subset.sort_values("importance_delta", ascending=False)
        if top_n is not None:
            subset = subset.head(top_n)

        subset["feature_pretty"] = subset["feature"].apply(_friendly_feature_name)
        sns.barplot(
            data=subset,
            x="feature_pretty",
            y="importance_delta",
            hue="feature_pretty",
            palette="mako",
            ax=ax,
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Permuted feature")
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for bar_idx, (_, row) in enumerate(subset.iterrows()):
            ax.text(
                bar_idx,
                row["importance_delta"],
                f"{row['importance_delta']:.3f}",
                ha="center",
                va="bottom" if row["importance_delta"] >= 0 else "top",
                fontsize=9,
                color="black",
            )

    fig.subplots_adjust(bottom=0.28, wspace=0.25)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


fig_pfi = plot_pfi_triptych(
    pfi_results,
    top_n=None,
    output_path=ARTIFACTS_DIR / "diagnostics" / "permutation_importance_all_tasks.png",
)
mlflow.log_figure(fig_pfi, "graficos/permutation_importance_all_tasks.png")
```

### Attention Maps

- Auxiliary model to extract attention (class/angle/energy heads).
- Average attention heatmaps with padding masks.



```python
# ===========================
# Auxiliary Model for Attention Scores Extraction
# ===========================
logging.info("Creating auxiliary model (CNN + RAW) to extract attention scores...")

# MHA layers (CNN)
mha_class_cnn = model.get_layer("mha_class_cnn")

# MHA layers (RAW)
mha_class_raw = model.get_layer("mha_class_raw")

# Normalized inputs (CNN)
pre_mha_class_cnn = model.get_layer("pre_mha_norm_class_cnn").output

# Normalized inputs (RAW)
pre_mha_class_raw = model.get_layer("pre_mha_norm_class_raw").output

# Calls with return_attention_scores=True
_, attn_class_cnn = mha_class_cnn(
    pre_mha_class_cnn, pre_mha_class_cnn, return_attention_scores=True
)
_, attn_class_raw = mha_class_raw(
    pre_mha_class_raw, pre_mha_class_raw, return_attention_scores=True
)

attention_model = Model(
    inputs=model.inputs,
    outputs=[
        attn_class_cnn,
        attn_class_raw,
    ],
    name="CONDOR_AttentionExtractor",
)

logging.info("Auxiliary attention model (CNN+RAW) created successfully.")
```


```python
# =======================================================
# Extracción y visualización: atención CNN vs atención RAW
sample_size = min(200, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test[sample_indices]
Xg_sample = Xg_test[sample_indices]

logging.info(f"Extracting attention scores from {sample_size} samples...")

# 1. Inferencia
with tf.device("/CPU:0"):
    (
        attn_class_cnn_scores,
        attn_class_raw_scores,
    ) = attention_model.predict([X_sample, Xg_sample], batch_size=4, verbose=1)

logging.info("Shapes CNN attention score (class): %s", attn_class_cnn_scores.shape)
logging.info("Shapes RAW attention score (class): %s", attn_class_raw_scores.shape)

# Máscaras de padding (longitud real en espacio RAW)
valid_masks_raw = ~np.all(np.isclose(X_sample, 0.0, atol=1e-8), axis=-1)
valid_lengths_raw = valid_masks_raw.sum(axis=-1)  # (batch,)


def average_attention(
    attn_scores: np.ndarray,
    valid_lengths: np.ndarray,
    max_len: int = 100,
    pool_factor: int = 1,
) -> np.ndarray:
    """
    Compute the average attention matrix across a batch, respecting padding.

    Accumulates attention matrices only over the valid (non-padded) region of each event
    and normalizes by the number of contributing events per position.
    """
    if attn_scores.ndim == 4:  # (batch, heads, L, L)
        attn_scores = attn_scores.mean(axis=1)

    batch, L_attn, _ = attn_scores.shape
    max_len = min(max_len, L_attn)
    attn_avg = np.zeros((max_len, max_len))
    count = np.zeros((max_len, max_len))

    for i in range(batch):
        L_eff = int(np.ceil(valid_lengths[i] / pool_factor))
        L_eff = max(0, min(L_eff, L_attn, max_len))
        if L_eff == 0:
            continue
        attn_avg[:L_eff, :L_eff] += attn_scores[i, :L_eff, :L_eff]
        count[:L_eff, :L_eff] += 1

    attn_avg = np.divide(attn_avg, count, where=count > 0)
    attn_avg[count == 0] = np.nan
    return attn_avg


# 2. Promedios (Directo y sin diccionarios multi-task)
avg_map_cnn = average_attention(
    attn_class_cnn_scores, valid_lengths_raw, max_len=80, pool_factor=2
)
avg_map_raw = average_attention(
    attn_class_raw_scores, valid_lengths_raw, max_len=120, pool_factor=1
)

# 3. Visualización 1x2 (Lado a lado) aplicando buenas prácticas (DRY)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# Configuraciones específicas para iterar y no repetir código de ploteo
plot_configs = [
    (axes[0], avg_map_cnn, "CNN branch", "Key pos (pooled)", "Query pos (pooled)"),
    (axes[1], avg_map_raw, "RAW branch", "Key pos (raw)", "Query pos (raw)"),
]

for ax, matrix, branch_name, xlabel, ylabel in plot_configs:
    vmax = float(np.nanpercentile(matrix, 99))
    sns.heatmap(
        matrix,
        cmap="viridis",
        square=True,
        ax=ax,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Attention Weight", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(
        f"Particle Classification\n{branch_name}\navg over heads & batch", fontsize=12
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.02,
        0.98,
        f"Max L: {matrix.shape[0]}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

# 4. Guardado y registro
plt.savefig(
    plots_dir / "attention_patterns_cnn_vs_raw.png", dpi=200, bbox_inches="tight"
)
plt.show()

logging.info(
    "Attention pattern plots saved to: %s",
    plots_dir / "attention_patterns_cnn_vs_raw.png",
)
mlflow.log_figure(fig, "graficos/attention_patterns_cnn_vs_raw.png")

```

### Attention vs Performance

- Compare attention for correct vs incorrect (or best/worst quartiles).
- Difference heatmaps and marginal distributions.



```python
# ----------------------------------------- #
# Análisis Avanzado de Patrones de Atención #
# ----------------------------------------- #
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

header_lines = [
    "=" * 60,
    "ANÁLISIS AVANZADO DE INTERPRETABILIDAD DE ATENCIÓN",
    "=" * 60,
]
logging.info("\n".join(header_lines))

# Usaremos las atenciones RAW (sin pooling) para evitar ambigüedad de longitudes
attn_class_scores = attn_class_raw_scores
valid_lengths = valid_lengths_raw  # longitudes reales en espacio RAW

# 1. ANÁLISIS POR PERFORMANCE: Predicciones correctas vs incorrectas
# ----------------------------- #
logging.info("\n1. Separando patrones por performance del modelo...")

# Definición de máscaras y etiquetas (Single-Task directo)
particle_correct_mask = pred_summary["particle_correct"].values[sample_indices]
good_mask = particle_correct_mask
bad_mask = ~particle_correct_mask
good_label = "Correct"
bad_label = "Incorrect"
task_name = "Particle Classification"

attn_3d = (
    attn_class_scores.mean(axis=1) if attn_class_scores.ndim == 4 else attn_class_scores
)
max_len = 60

# --- Patrón promedio para predicciones BUENAS ---
attn_good = np.zeros((max_len, max_len))
count_good = np.zeros((max_len, max_len))
for idx in np.where(good_mask)[0]:
    L = min(int(valid_lengths[idx]), max_len)
    if L > 0:
        attn_good[:L, :L] += attn_3d[idx, :L, :L]
        count_good[:L, :L] += 1
attn_good = np.divide(attn_good, count_good, where=count_good > 0)
attn_good[count_good == 0] = np.nan

# --- Patrón promedio para predicciones MALAS ---
attn_bad = np.zeros((max_len, max_len))
count_bad = np.zeros((max_len, max_len))
for idx in np.where(bad_mask)[0]:
    L = min(int(valid_lengths[idx]), max_len)
    if L > 0:
        attn_bad[:L, :L] += attn_3d[idx, :L, :L]
        count_bad[:L, :L] += 1
attn_bad = np.divide(attn_bad, count_bad, where=count_bad > 0)
attn_bad[count_bad == 0] = np.nan

# --- DIFERENCIA entre buenos y malos ---
attn_diff = attn_good - attn_bad

# VISUALIZACIÓN LADO A LADO (1 fila x 4 columnas)
fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

# Plot 1: Predicciones BUENAS
ax1 = axes[0]
vmax_good = (
    float(np.nanpercentile(attn_good, 99)) if np.isfinite(attn_good).any() else 1.0
)
sns.heatmap(
    attn_good,
    cmap="viridis",
    square=True,
    cbar=True,
    ax=ax1,
    vmin=0,
    vmax=vmax_good,
    cbar_kws={"shrink": 0.7},
    xticklabels=False,
    yticklabels=False,
)
ax1.set_title(
    f"{task_name}\n{good_label} (n={good_mask.sum()})", fontweight="bold", fontsize=11
)
ax1.set_ylabel("Query Position", fontsize=9)

# Plot 2: Predicciones MALAS
ax2 = axes[1]
vmax_bad = float(np.nanpercentile(attn_bad, 99)) if np.isfinite(attn_bad).any() else 1.0
sns.heatmap(
    attn_bad,
    cmap="viridis",
    square=True,
    cbar=True,
    ax=ax2,
    vmin=0,
    vmax=vmax_bad,
    cbar_kws={"shrink": 0.7},
    xticklabels=False,
    yticklabels=False,
)
ax2.set_title(
    f"{task_name}\n{bad_label} (n={bad_mask.sum()})", fontweight="bold", fontsize=11
)

# Plot 3: DIFERENCIA (Good - Bad)
ax3 = axes[2]
vmax_diff = (
    float(np.nanpercentile(np.abs(attn_diff), 95))
    if np.isfinite(attn_diff).any()
    else 1.0
)
sns.heatmap(
    attn_diff,
    cmap="RdBu_r",
    center=0,
    square=True,
    cbar=True,
    ax=ax3,
    vmin=-vmax_diff,
    vmax=vmax_diff,
    cbar_kws={"shrink": 0.7, "label": "Δ Attention"},
    xticklabels=False,
    yticklabels=False,
)
ax3.set_title(
    f"Difference\n({good_label} - {bad_label})", fontweight="bold", fontsize=11
)

# Plot 4: Distribución de atención por posición (marginal)
ax4 = axes[3]
attn_received_good = np.nanmean(attn_good, axis=0)[:max_len]
attn_received_bad = np.nanmean(attn_bad, axis=0)[:max_len]
positions = np.arange(len(attn_received_good))

ax4.plot(
    positions,
    attn_received_good,
    label=good_label,
    color="green",
    linewidth=2,
    alpha=0.7,
)
ax4.plot(
    positions, attn_received_bad, label=bad_label, color="red", linewidth=2, alpha=0.7
)
ax4.fill_between(positions, attn_received_good, alpha=0.2, color="green")
ax4.fill_between(positions, attn_received_bad, alpha=0.2, color="red")
ax4.set_xlabel("Temporal Position", fontsize=9)
ax4.set_ylabel("Avg Attention Received", fontsize=9)
ax4.set_title("Attention Distribution", fontweight="bold", fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.suptitle(
    "Attention Patterns: Performance-Based Analysis",
    fontsize=16,
    fontweight="bold",
    y=1.05,
)
plt.savefig(
    plots_dir / "attention_performance_comparison.png", dpi=200, bbox_inches="tight"
)
plt.show()

logging.info("✓ Guardado: %s", plots_dir / "attention_performance_comparison.png")
mlflow.log_figure(fig, "graficos/attention_performance_comparison.png")

```

### Temporal Attention Patterns

- Early/Mid/Late attention distributions.
- Attention entropy; self vs context (diagonal/off-diagonal).



```python
# =======================================================
# 2. ANÁLISIS TEMPORAL DE ATENCIÓN

logging.info("\n2. Analizando importancia temporal de posiciones...")

# Single-task: únicamente clasificación Gamma/Hadrón
attn_scores = attn_class_raw_scores

attn_3d = attn_scores.mean(axis=1) if attn_scores.ndim == 4 else attn_scores

max_len = 80

early_range = (0, max_len // 3)
mid_range = (max_len // 3, 2 * max_len // 3)
late_range = (2 * max_len // 3, max_len)

# Inicializamos listas
early_attn, mid_attn, late_attn = [], [], []
entropies = []
diag_strengths, offdiag_strengths = [], []

# -------------------------------------------------------
# Procesamiento (Loop unificado de alta eficiencia)
# -------------------------------------------------------
for idx in range(len(attn_3d)):
    L = min(int(valid_lengths[idx]), max_len)

    if L > 5:
        attn_mat = attn_3d[idx, :L, :L]

        # --- 1. Extracción Temporal (Solo si L > 10) ---
        if L > 10:
            attn_received = attn_mat.mean(axis=0)

            e_end = min(L, early_range[1])
            m_start = min(L, mid_range[0])
            m_end = min(L, mid_range[1])
            l_start = min(L, late_range[0])

            if e_end > early_range[0]:
                early_attn.append(np.nanmean(attn_received[early_range[0] : e_end]))
            if m_end > m_start:
                mid_attn.append(np.nanmean(attn_received[m_start:m_end]))
            if L > l_start:
                late_attn.append(np.nanmean(attn_received[l_start:L]))

        # --- 2. Entropía ---
        for row in attn_mat:
            s = row.sum()
            if s > 0:
                row_norm = row / s
                entropy = -np.sum(row_norm * np.log(row_norm + 1e-12))
                if np.isfinite(entropy):
                    entropies.append(entropy)

        # --- 3. Self vs Context ---
        diag = np.diag(attn_mat).mean()
        mask = ~np.eye(L, dtype=bool)
        offdiag = attn_mat[mask].mean()

        if np.isfinite(diag) and np.isfinite(offdiag):
            diag_strengths.append(diag)
            offdiag_strengths.append(offdiag)


# VISUALIZACIÓN 1x3 (Single-Task)
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    constrained_layout=True,
)

task_name = "Gamma-Hadron Discrimination"

# -------------------------------------------------------
# Plot 1: Box plot (Importancia Temporal)
# -------------------------------------------------------
ax1 = axes[0]
bp = ax1.boxplot(
    [early_attn, mid_attn, late_attn],
    labels=["Early", "Mid", "Late"],
    patch_artist=True,
)
for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen", "lightcoral"]):
    patch.set_facecolor(color)

ax1.set_title(
    f"{task_name}\nAttention by Temporal Region", fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Avg Attention Weight", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# -------------------------------------------------------
# Plot 2: Distribución de Entropía
# -------------------------------------------------------
ax2 = axes[1]
if entropies:
    ax2.hist(entropies, bins=50, color="purple", alpha=0.6, edgecolor="black")
    mean_ent = float(np.mean(entropies))
    ax2.axvline(
        mean_ent,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_ent:.3f}",
    )
    ax2.legend(fontsize=8)

ax2.set_title("Attention Entropy Distribution", fontsize=11, fontweight="bold")
ax2.set_xlabel("Entropy (higher = more diffuse)", fontsize=9)
ax2.set_ylabel("Frequency", fontsize=9)
ax2.grid(alpha=0.3)

# -------------------------------------------------------
# Plot 3: Self vs Context (Diagonal vs Off-diagonal)
# -------------------------------------------------------
ax3 = axes[2]
if diag_strengths and offdiag_strengths:
    ax3.scatter(offdiag_strengths, diag_strengths, alpha=0.3, s=10, color="navy")
    # Cálculo seguro del límite para el gráfico
    lim = float(max(max(offdiag_strengths), max(diag_strengths)))
    ax3.plot(
        [0, lim],
        [0, lim],
        "r--",
        linewidth=2,
        alpha=0.5,
        label="Diagonal = Off-diagonal",
    )
    ax3.legend(fontsize=8)

ax3.set_xlabel("Off-diagonal Attention (Context)", fontsize=9)
ax3.set_ylabel("Diagonal Attention (Self)", fontsize=9)
ax3.set_title("Self-Attention vs Context", fontsize=11, fontweight="bold")
ax3.grid(alpha=0.3)

# -------------------------------------------------------
# Guardado y registro
# -------------------------------------------------------
plt.savefig(plots_dir / "attention_temporal_analysis.png", dpi=200, bbox_inches="tight")
plt.show()

logging.info("✓ Guardado: %s", plots_dir / "attention_temporal_analysis.png")
mlflow.log_figure(fig, "graficos/attention_temporal_analysis.png")

```

### Attention by Physics (Particle & Energy)

- Attention profiles split by particle type and energy level.
- Aggregated heatmaps by energy bin.



```python
# 3. ANÁLISIS POR TIPO DE PARTÍCULA Y ENERGÍA (Gamma-Hadron Classifier)
# -------------------------------------------------------------------- #
logging.info("\n3. Analizando patrones por tipo de partícula y energía...")

particle_types = pred_summary["particle_true"].values[sample_indices]
energy_levels_sample = pred_summary["energy_level_true"].values[sample_indices]

task_name = "Particle Classification"
attn_scores = attn_class_scores
attn_3d = attn_scores.mean(axis=1) if attn_scores.ndim == 4 else attn_scores

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# --- Plot 1: Atención por tipo de partícula ---
ax1 = axes[0]
max_plot_len = 60
for particle_type, color, label in [
    (0, "blue", "Proton"),
    (1, "orange", r"$\gamma$"),
]:
    mask = particle_types == particle_type
    if mask.sum() > 0:
        attention_by_pos = np.full((mask.sum(), max_plot_len), np.nan)
        valid_count = 0
        for array_idx, idx in enumerate(np.where(mask)[0]):
            L = min(int(valid_lengths[idx]), max_plot_len)
            if L > 5:
                attn_mat = attn_3d[idx, :L, :L]
                attention_by_pos[valid_count, :L] = attn_mat.mean(axis=0)
                valid_count += 1
        if valid_count > 0:
            avg_attn = np.nanmean(attention_by_pos[:valid_count], axis=0)
            positions = np.arange(len(avg_attn))
            ax1.plot(
                positions, avg_attn, label=label, color=color, linewidth=2, alpha=0.7
            )
            ax1.fill_between(positions, avg_attn, alpha=0.2, color=color)
ax1.set_title(
    f"{task_name}\nAttention by Particle Type", fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Temporal Position", fontsize=9)
ax1.set_ylabel("Avg Attention", fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# --- Plot 2: Atención por nivel de energía (covariable física) ---
ax2 = axes[1]
energy_colors = {"3E2": "green", "5E2": "orange", "8E2": "red"}
energy_labels = {"3E2": "300 GeV", "5E2": "500 GeV", "8E2": "800 GeV"}
for energy_level in ["3E2", "5E2", "8E2"]:
    mask = energy_levels_sample == energy_level
    if mask.sum() > 0:
        attention_by_pos = np.full((mask.sum(), max_plot_len), np.nan)
        valid_count = 0
        for array_idx, idx in enumerate(np.where(mask)[0]):
            L = min(int(valid_lengths[idx]), max_plot_len)
            if L > 5:
                attn_mat = attn_3d[idx, :L, :L]
                attention_by_pos[valid_count, :L] = attn_mat.mean(axis=0)
                valid_count += 1
        if valid_count > 0:
            avg_attn = np.nanmean(attention_by_pos[:valid_count], axis=0)
            positions = np.arange(len(avg_attn))
            ax2.plot(
                positions,
                avg_attn,
                label=energy_labels[energy_level],
                color=energy_colors[energy_level],
                linewidth=2,
                alpha=0.7,
            )
ax2.set_title("Attention by Energy Level", fontsize=11, fontweight="bold")
ax2.set_xlabel("Temporal Position", fontsize=9)
ax2.set_ylabel("Avg Attention", fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# --- Plot 3: Heatmap agregado por energía ---
ax3 = axes[2]
energy_attn_summary = []
energy_labels_list = []
max_heatmap_len = 40
for energy_level in ["3E2", "5E2", "8E2"]:
    mask = energy_levels_sample == energy_level
    if mask.sum() > 0:
        attn_avg = np.zeros((max_heatmap_len, max_heatmap_len))
        count = np.zeros((max_heatmap_len, max_heatmap_len))
        for idx in np.where(mask)[0]:
            L = min(int(valid_lengths[idx]), max_heatmap_len)
            if L > 5:
                attn_avg[:L, :L] += attn_3d[idx, :L, :L]
                count[:L, :L] += 1
        attn_avg = np.divide(attn_avg, count, where=count > 0)
        attn_avg[count == 0] = np.nan
        row_means = []
        for i in range(max_heatmap_len):
            if count[i, :].sum() > 0:
                valid_vals = attn_avg[i, count[i, :] > 0]
                row_means.append(
                    np.nanmean(valid_vals) if len(valid_vals) > 0 else np.nan
                )
            else:
                row_means.append(np.nan)
        energy_attn_summary.append(row_means)
        energy_labels_list.append(energy_labels[energy_level])
if energy_attn_summary:
    im = ax3.imshow(
        energy_attn_summary, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax3.set_yticks(range(len(energy_labels_list)))
    ax3.set_yticklabels(energy_labels_list)
    ax3.set_xlabel("Query Position", fontsize=9)
    ax3.set_title("Attention Heatmap by Energy", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax3, shrink=0.8, label="Attention")

plt.savefig(plots_dir / "attention_physics_analysis.png", dpi=200, bbox_inches="tight")
plt.show()
logging.info("Guardado: %s", plots_dir / "attention_physics_analysis.png")
mlflow.log_figure(fig, "graficos/attention_physics_analysis.png")

completed_lines = [
    "",
    "=" * 60,
    "ANÁLISIS COMPLETADO",
    "=" * 60,
    "",
    "Resumen de visualizaciones generadas:",
    "1. attention_performance_comparison.png - Patrones según performance",
    "2. attention_temporal_analysis.png - Análisis temporal y entropía",
    "3. attention_physics_analysis.png - Análisis por física del evento",
]
logging.info("\n".join(completed_lines))

```

### Spatial Attention (Detector Map)

- Attention aggregated per detector_id using geometry.
- Radial trends, quadrant analysis.



```python
# ============================================================
# ANÁLISIS ESPACIAL: ¿Qué regiones del detector son importantes?
# Contexto: CONDOR EAS — Features: detector_id, particle_count,
#   t_bin, total_energy, x_center, y_center
# Arquitectura: CNN → Transformer → head clasificación gamma-hadrón
# ============================================================
logging.info("\n[1] ANÁLISIS ESPACIAL: Mapeo de importancia por detector")

detector_positions = detector_catalog.set_index("detector_id")[
    ["x_center", "y_center"]
].to_dict("index")

task_name = "Particle Classification"
attn_scores = attn_class_scores
attn_3d = attn_scores.mean(axis=1) if attn_scores.ndim == 4 else attn_scores

fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# --- Acumular importancia por detector ---
detector_importance = {}
detector_count = {}
for idx in range(len(X_sample)):
    seq = X_sample[idx]
    L = int(valid_lengths[idx])
    if L <= 5:
        continue
    attn_mat = attn_3d[idx]
    attn_len = attn_mat.shape[0]
    pool_factor = max(1, int(np.ceil(L / max(1, attn_len))))
    attn_received = attn_mat.mean(axis=0)
    attn_received = np.where(np.isfinite(attn_received), attn_received, np.nan)
    if attn_received.size == 0:
        continue
    max_t = min(attn_len, int(np.ceil(L / pool_factor)))
    for t_pos in range(max_t):
        seq_idx = min(int(t_pos * pool_factor), L - 1)
        det_id = int(seq[seq_idx, 0])
        if det_id >= 0 and det_id in detector_positions:
            if det_id not in detector_importance:
                detector_importance[det_id] = 0.0
                detector_count[det_id] = 0
            if t_pos < len(attn_received):
                val = attn_received[t_pos]
                if np.isfinite(val):
                    detector_importance[det_id] += val
                    detector_count[det_id] += 1
for det_id in list(detector_importance):
    if detector_count[det_id] > 0:
        detector_importance[det_id] /= detector_count[det_id]
    else:
        detector_importance.pop(det_id, None)

if len(detector_importance) == 0:
    logging.warning("No hay datos de importancia de detectores para %s", task_name)
else:
    detector_ids = list(detector_importance.keys())
    x_coords = [detector_positions[d]["x_center"] for d in detector_ids]
    y_coords = [detector_positions[d]["y_center"] for d in detector_ids]
    importances = np.array([detector_importance[d] for d in detector_ids], dtype=float)

    logging.info(
        "\n%s:\n  Detectores analizados: %d\n  Atención promedio: %.6f ± %.6f",
        task_name,
        len(detector_ids),
        np.nanmean(importances),
        np.nanstd(importances),
    )

    # --- Plot 1: Mapa espacial de atención ---
    scatter = axes[0].scatter(
        x_coords,
        y_coords,
        c=importances,
        s=100,
        cmap="hot",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("X Position (m)", fontsize=11)
    axes[0].set_ylabel("Y Position (m)", fontsize=11)
    axes[0].set_title(
        f"{task_name}\nSpatial Attention Distribution",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].grid(alpha=0.3)
    axes[0].set_aspect("equal")
    plt.colorbar(scatter, ax=axes[0]).set_label("Avg Attention", fontsize=10)

    # --- Plot 2: Atención vs distancia radial ---
    distances = np.sqrt(np.array(x_coords) ** 2 + np.array(y_coords) ** 2)
    axes[1].scatter(distances, importances, alpha=0.5, s=30, color="navy")
    if len(distances) > 3:
        z = np.polyfit(distances, importances, 2)
        x_fit = np.linspace(distances.min(), distances.max(), 100)
        axes[1].plot(
            x_fit, np.poly1d(z)(x_fit), "r--", linewidth=2, label="Quadratic fit"
        )
    corr, pval = (
        pearsonr(distances, importances) if len(distances) > 1 else (np.nan, np.nan)
    )
    axes[1].set_xlabel("Distance from Center (m)", fontsize=11)
    axes[1].set_ylabel("Avg Attention", fontsize=11)
    axes[1].set_title(
        f"Attention vs Radial Distance\nr={corr:.3f}, p={pval:.2e}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # --- Plot 3: Atención por cuadrante ---
    quadrants = {"Q1 (+,+)": [], "Q2 (-,+)": [], "Q3 (-,-)": [], "Q4 (+,-)": []}
    for x, y, imp in zip(x_coords, y_coords, importances):
        if x >= 0 and y >= 0:
            quadrants["Q1 (+,+)"].append(imp)
        elif x < 0 and y >= 0:
            quadrants["Q2 (-,+)"].append(imp)
        elif x < 0 and y < 0:
            quadrants["Q3 (-,-)"].append(imp)
        else:
            quadrants["Q4 (+,-)"].append(imp)
    quad_means = [
        np.nanmean(quadrants[q]) if len(quadrants[q]) > 0 else 0
        for q in ["Q1 (+,+)", "Q2 (-,+)", "Q3 (-,-)", "Q4 (+,-)"]
    ]
    colors_quad = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    bars = axes[2].bar(
        range(4), quad_means, color=colors_quad, edgecolor="black", linewidth=1.5
    )
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["Q1\n(+,+)", "Q2\n(-,+)", "Q3\n(-,-)", "Q4\n(+,-)"])
    axes[2].set_ylabel("Mean Attention", fontsize=11)
    axes[2].set_title("Attention by Quadrant", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig(
        plots_dir / "attention_spatial_analysis.png", dpi=200, bbox_inches="tight"
    )
    plt.show()
    logging.info("Guardado: %s", plots_dir / "attention_spatial_analysis.png")
    mlflow.log_figure(fig, "graficos/attention_spatial_analysis.png")

```

### Correlation: Attention vs Physical Features

- Spearman correlations with particle_count, total_energy, temporal position, radial distance.



```python
# ============================================================
# CORRELACIÓN ATENCIÓN - FEATURES FÍSICAS
# ============================================================
logging.info("\n[3] CORRELACIÓN ENTRE ATENCIÓN Y FEATURES FÍSICAS")

task_name = "Particle Classification"
attn_scores = attn_class_scores
attn_3d = attn_scores.mean(axis=1) if attn_scores.ndim == 4 else attn_scores

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# --- Acumular pares (atención, feature) hit a hit ---
attention_weights = []
particle_counts = []
energies = []

for idx in range(len(X_sample)):
    seq = X_sample[idx]
    L = int(valid_lengths[idx])
    if L <= 5:
        continue
    attn_mat = attn_3d[idx]
    attn_len = attn_mat.shape[0]
    if attn_len == 0:
        continue
    pool_factor = max(1, int(np.round(L / max(1, attn_len))))
    attn_received = attn_mat.mean(axis=0)
    attn_received = np.where(np.isfinite(attn_received), attn_received, np.nan)
    max_t = min(attn_len, attn_received.shape[0])
    for t_pos in range(max_t):
        seq_idx = min(int(t_pos * pool_factor), L - 1)
        det_id = int(seq[seq_idx, 0])
        if det_id >= 0 and det_id in detector_positions:
            aw = attn_received[t_pos]
            if not np.isfinite(aw):
                continue
            attention_weights.append(float(aw))
            particle_counts.append(float(seq[seq_idx, 1]))
            energies.append(float(seq[seq_idx, 3]))

if len(attention_weights) == 0:
    logging.warning(
        "No hay datos para correlación de atención-features en %s", task_name
    )
else:
    attention_weights = np.array(attention_weights, dtype=float)
    particle_counts = np.array(particle_counts, dtype=float)
    energies = np.array(energies, dtype=float)

    # --- Plot 1: Atención vs Particle Count ---
    axes[0].hexbin(
        particle_counts, attention_weights, gridsize=30, cmap="YlOrRd", mincnt=1
    )
    corr_pc, pval_pc = (
        spearmanr(particle_counts, attention_weights)
        if len(particle_counts) > 2
        else (np.nan, np.nan)
    )
    axes[0].set_xlabel("Particle Count", fontsize=11)
    axes[0].set_ylabel("Attention Weight", fontsize=11)
    axes[0].set_title(
        f"{task_name}\nAttention vs Particle Count\nρ={corr_pc:.3f}, p={pval_pc:.2e}",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].grid(alpha=0.3)

    # --- Plot 2: Atención vs Total Energy ---
    axes[1].hexbin(energies, attention_weights, gridsize=30, cmap="viridis", mincnt=1)
    corr_e, pval_e = (
        spearmanr(energies, attention_weights)
        if len(energies) > 2
        else (np.nan, np.nan)
    )
    axes[1].set_xlabel("Total Energy", fontsize=11)
    axes[1].set_ylabel("Attention Weight", fontsize=11)
    axes[1].set_title(
        f"Attention vs Total Energy\nρ={corr_e:.3f}, p={pval_e:.2e}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].grid(alpha=0.3)

    plt.savefig(
        plots_dir / "attention_feature_correlation.png", dpi=200, bbox_inches="tight"
    )
    plt.show()
    logging.info("Guardado: %s", plots_dir / "attention_feature_correlation.png")
    mlflow.log_figure(fig, "graficos/attention_feature_correlation.png")

```

### CNN Embedding Space

- PCA on pooled CNN embeddings.
- Color by particle type, energy, correctness.



```python
# ============================================================
# 4. ANÁLISIS DE EMBEDDING: Representación CNN
# ============================================================
logging.info("\n[4] ANÁLISIS DE EMBEDDING: ¿Qué aprende la CNN?")

# Crear modelo intermedio para extraer embeddings CNN
cnn_output_layer = model.get_layer("dense_after_cnn4")
embedding_model = Model(inputs=model.inputs, outputs=cnn_output_layer.output)

# Extraer embeddings para subset
logging.info("Extrayendo embeddings CNN para %d muestras...", len(X_sample))
embeddings = embedding_model.predict([X_sample, Xg_sample], batch_size=64, verbose=0)
# embeddings shape: (batch, seq_len, 128)

# Promediar sobre secuencia para obtener representación global (evitar NaN si L=0)
embeddings_avg_list = []
for i, emb in enumerate(embeddings):
    L = int(valid_lengths[i])
    if L <= 0:
        embeddings_avg_list.append(np.zeros(emb.shape[-1], dtype=np.float32))
    else:
        vec = np.nanmean(emb[:L], axis=0)
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings_avg_list.append(vec.astype(np.float32))
embeddings_avg = np.vstack(embeddings_avg_list)

# PCA para visualización (salir si hay menos de 2 muestras)
if embeddings_avg.shape[0] < 2:
    logging.warning(
        "No hay suficientes embeddings para PCA (n=%d)", embeddings_avg.shape
    )
else:
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_avg)
    particle_types = pred_summary["particle_true"].values[sample_indices]
    energy_levels_sample = pred_summary["energy_level_true"].values[sample_indices]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Colored by particle type
    scatter1 = axes[0].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=particle_types,
        cmap="coolwarm",
        s=30,
        alpha=0.6,
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=11)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=11)
    axes[0].set_title(
        "CNN Embedding Space\nColored by Particle Type", fontsize=12, fontweight="bold"
    )
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Particle Type", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2: Colored by energy
    energy_numeric = np.array(
        [float(e.replace("E", "e")) for e in energy_levels_sample]
    )
    scatter2 = axes[1].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=energy_numeric,
        cmap="plasma",
        s=30,
        alpha=0.6,
    )
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=11)
    axes[1].set_title(
        "CNN Embedding Space\nColored by Energy", fontsize=12, fontweight="bold"
    )
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Energy (GeV)", fontsize=10)
    axes[1].grid(alpha=0.3)

    # Plot 3: Colored by prediction correctness
    particle_correct = particle_correct_mask.astype(int)
    scatter3 = axes[2].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=particle_correct,
        cmap="RdYlGn",
        s=30,
        alpha=0.6,
    )
    axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=11)
    axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=11)
    axes[2].set_title(
        "CNN Embedding Space\nColored by Prediction Correctness",
        fontsize=12,
        fontweight="bold",
    )
    cbar3 = plt.colorbar(scatter3, ax=axes[2])
    cbar3.set_label("Correct Prediction", fontsize=10)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "cnn_embedding_analysis.png", dpi=200, bbox_inches="tight")
    plt.show()

    logging.info("✓ Guardado: %s", plots_dir / "cnn_embedding_analysis.png")
mlflow.log_figure(fig, "graficos/cnn_embedding_analysis.png")
```

### Atención: curva global y resumen simple (RAW)



```python
# --- Atención: curva global y resumen simple (RAW) ---
def attention_received_per_event(
    attn_scores: np.ndarray,
    valid_lengths: np.ndarray,
    max_len: int = 120,
) -> list[np.ndarray]:
    """
    Extract the attention received profile for each event in the batch.

    For each event, computes the mean attention weight received by each key position (column mean of the attention matrix), restricted to the valid non-padded region. This profile indicates which temporal positions in the hit sequence were most attended to by the model.

    Parameters
    ----------
    attn_scores : np.ndarray
        Attention score array of shape (batch, heads, L, L) or (batch, L, L). If 4D, the head dimension is averaged before processing.
    valid_lengths : np.ndarray
        Integer array of shape (batch,) with the unpadded sequence length for each event.
    max_len : int, optional
        Maximum sequence length to consider. Default is 120.

    Returns
    -------
    list[np.ndarray]
        List of 1D arrays, one per valid event (events with L <= 0 are skipped). Each array has length min(L, max_len) and contains the mean attention weight received at each temporal position.
    """
    if attn_scores.ndim == 4:
        attn_scores = attn_scores.mean(axis=1)  # heads -> mean
    curves = []
    for mat, L in zip(attn_scores, valid_lengths):
        L = int(min(L, max_len, mat.shape[0]))
        if L <= 0:
            continue
        attn_recv = mat[:L, :L].mean(axis=0)
        curves.append(attn_recv)
    return curves


def plot_global_attention(
    attn_scores: np.ndarray,
    valid_lengths: np.ndarray,
    title: str,
    save_path: Path,
    max_len: int = 120,
) -> None:
    """
    Plot a three-panel summary of global attention patterns across the batch.

    Produces a figure with three panels:
    - Left  : heatmap of the attention received profile for a single example event (first valid event in the batch).
    - Center: mean ± 1σ attention received curve across all valid events, showing which temporal positions are consistently attended to.
    - Right : bar chart comparing mean attention in Early, Mid, and Late thirds of the sequence, summarizing the temporal attention bias.

    Parameters
    ----------
    attn_scores : np.ndarray
        Attention score array of shape (batch, heads, L, L) or (batch, L, L).
    valid_lengths : np.ndarray
        Integer array of shape (batch,) with unpadded sequence lengths.
    title : str
        Figure super-title, also used in the warning log if no valid events are found.
    save_path : Path
        Destination path for saving the figure at 200 dpi.
    max_len : int, optional
        Maximum sequence length to include in the analysis. Default is 120.

    Returns
    -------
    None
        Figure is saved to save_path and rendered via plt.show(). Logs a warning and returns early if no valid events are found.
    """
    curves = attention_received_per_event(attn_scores, valid_lengths, max_len)
    if not curves:
        logging.warning("No hay muestras válidas para atención en %s", title)
        return
    max_L = min(max(len(c) for c in curves), max_len)
    padded = []
    for c in curves:
        Lc = min(len(c), max_L)
        pad = np.full(max_L, np.nan, dtype=float)
        pad[:Lc] = c[:Lc]
        padded.append(pad)
    stack = np.vstack(padded)
    mean_curve = np.nanmean(stack, axis=0)
    std_curve = np.nanstd(stack, axis=0)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 4), gridspec_kw={"width_ratios": [1, 1.2, 1]}
    )
    sns.heatmap(
        curves[0][:, None],
        cmap="viridis",
        ax=axes[0],
        cbar=True,
        yticklabels=False,
        xticklabels=False,
    )
    axes[0].set_title("Ejemplo: atención recibida\npor posición (key)")
    x = np.arange(max_L)
    axes[1].plot(x, mean_curve, color="tab:blue", label="media")
    axes[1].fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="tab:blue",
        alpha=0.2,
        label="±1σ",
    )
    axes[1].set_xlabel("Posición temporal (key)")
    axes[1].set_ylabel("Atención recibida")
    axes[1].set_title("Curva global (media ± σ)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    thirds = np.array_split(mean_curve, 3)
    parts = [np.nanmean(t) for t in thirds]
    axes[2].bar(
        ["Early", "Mid", "Late"],
        parts,
        color=["#4C9AFF", "#36CFC9", "#FF8A65"],
        edgecolor="k",
    )
    axes[2].set_ylim(0, np.nanmax(parts) * 1.2)
    axes[2].set_ylabel("Atención media")
    axes[2].set_title("Distribución Early/Mid/Late")
    for i, v in enumerate(parts):
        axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    logging.info("✓ Guardado: %s", save_path)


plot_global_attention(
    attn_class_raw_scores,
    valid_lengths_raw,
    "Attention (RAW) - Particle head",
    plots_dir / "attn_global_particle.png",
)
mlflow.log_figure(fig, "graficos/attn_global_particle.png")

```

### Top-k hits más influyentes por evento (RAW)



```python
# --- Top-k hits más influyentes por evento (RAW) ---

FEATURE_IDX = {
    "detector_id": 0,
    "particle_count": 1,
    "t_bin": 2,
    "total_energy": 3,
    "x_center": 4,
    "y_center": 5,
}


def topk_attention_hits(
    attn_scores: np.ndarray,
    sequences: np.ndarray,
    valid_lengths: np.ndarray,
    k: int = 5,
) -> list[pd.DataFrame]:
    """
    Identify the k detector hits with highest received attention per event.

    For each event, ranks hits by the attention weight they receive (column mean of the attention matrix) and returns the top-k hits with their full feature values. This provides a physically interpretable view of which specific detector activations the model focuses on when making its predictions.

    Parameters
    ----------
    attn_scores : np.ndarray
        Attention score array of shape (batch, heads, L, L) or (batch, L, L). If 4D, the head dimension is averaged before ranking.
    sequences : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features), used to retrieve feature values for the top-k hits.
    valid_lengths : np.ndarray
        Integer array of shape (batch,) with unpadded sequence lengths.
    k : int, optional
        Number of top hits to return per event. Default is 5.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per event. Each DataFrame has k rows and columns: rank, attention, detector_id, t_bin, particle_count, total_energy, x_center, y_center. Events with L <= 0 are represented by an empty DataFrame.
    """
    if attn_scores.ndim == 4:
        attn_scores = attn_scores.mean(axis=1)
    results = []
    for mat, seq, L in zip(attn_scores, sequences, valid_lengths):
        L = int(min(L, mat.shape[0], seq.shape[0]))
        if L <= 0:
            results.append(pd.DataFrame())
            continue
        attn_recv = mat[:L, :L].mean(axis=0)
        top_idx = np.argsort(attn_recv)[::-1][:k]
        rows = []
        for idx in top_idx:
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "attention": float(attn_recv[idx]),
                    "detector_id": int(seq[idx, FEATURE_IDX["detector_id"]]),
                    "t_bin": float(seq[idx, FEATURE_IDX["t_bin"]]),
                    "particle_count": float(seq[idx, FEATURE_IDX["particle_count"]]),
                    "total_energy": float(seq[idx, FEATURE_IDX["total_energy"]]),
                    "x_center": float(seq[idx, FEATURE_IDX["x_center"]]),
                    "y_center": float(seq[idx, FEATURE_IDX["y_center"]]),
                }
            )
        results.append(pd.DataFrame(rows))
    return results


topk_particle = topk_attention_hits(
    attn_class_raw_scores, X_sample, valid_lengths_raw, k=5
)

# Ejemplo: mostrar primer evento y guardar CSV
if topk_particle and not topk_particle[0].empty:
    display(topk_particle[0])
all_topk = pd.concat(topk_particle, keys=range(len(topk_particle)), names=["event_idx"])
all_topk.to_csv(plots_dir / "topk_attention_hits_particle.csv")
logging.info("✓ Guardado: %s", plots_dir / "topk_attention_hits_particle.csv")
```

### Mapa espacial: atención media por detector (RAW, head de partícula)



```python
# --- Mapa espacial sencillo: atención media por detector (RAW, head de partícula) ---
def spatial_attention_map(
    attn_scores: np.ndarray,
    sequences: np.ndarray,
    valid_lengths: np.ndarray,
    detector_catalog: pd.DataFrame,
    title: str,
    save_path: Path,
    max_len: int = 120,
) -> None:
    """
    Plot the mean received attention projected onto the physical detector array.

    Aggregates attention weights by detector_id across all events and time steps, then renders the result as a scatter plot over the (x_center, y_center) geometry of the CONDOR array. This reveals whether the model preferentially attends to central or peripheral detectors, and whether attention exhibits spatial asymmetries correlated with shower direction.

    Parameters
    ----------
    attn_scores : np.ndarray
        Attention score array of shape (batch, heads, L, L) or (batch, L, L). If 4D, the head dimension is averaged before accumulation.
    sequences : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features). Column 0 must contain detector_id values matching detector_catalog.
    valid_lengths : np.ndarray
        Integer array of shape (batch,) with unpadded sequence lengths.
    detector_catalog : pd.DataFrame
        Detector geometry table with columns: detector_id, x_center, y_center. Produced by build_detector_catalog().
    title : str
        Plot title, also used in the warning log if no data is available.
    save_path : Path
        Destination path for saving the figure at 200 dpi.
    max_len : int, optional
        Maximum sequence length to process per event. Default is 120.

    Returns
    -------
    None
        Figure is saved to save_path and rendered via plt.show(). Logs a warning and returns early if no valid attention data is found.
    """
    if attn_scores.ndim == 4:
        attn_scores = attn_scores.mean(axis=1)
    pos = detector_catalog.set_index("detector_id")[["x_center", "y_center"]]
    acc = {}
    cnt = {}
    for mat, seq, L in zip(attn_scores, sequences, valid_lengths):
        L = int(min(L, mat.shape[0], seq.shape[0], max_len))
        if L <= 0:
            continue
        attn_recv = mat[:L, :L].mean(axis=0)
        for t in range(L):
            det = int(seq[t, FEATURE_IDX["detector_id"]])
            if det in pos.index and np.isfinite(attn_recv[t]):
                acc[det] = acc.get(det, 0.0) + float(attn_recv[t])
                cnt[det] = cnt.get(det, 0) + 1
    if not acc:
        logging.warning("No hay datos para mapa espacial en %s", title)
        return
    imp = {d: acc[d] / cnt[d] for d in acc}
    df = pd.DataFrame(
        {"detector_id": list(imp.keys()), "attention": list(imp.values())}
    ).merge(pos, left_on="detector_id", right_index=True)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        df.x_center,
        df.y_center,
        c=df.attention,
        s=120,
        cmap="plasma",
        edgecolor="k",
        linewidth=0.6,
    )
    plt.colorbar(sc, label="Atención media")
    plt.xlabel("x_center (m)")
    plt.ylabel("y_center (m)")
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    logging.info("✓ Guardado: %s", save_path)


# --- Llamada: head de clasificación gamma-hadrón (RAW) ---
spatial_attention_map(
    attn_class_raw_scores,
    X_sample,
    valid_lengths_raw,
    detector_catalog,
    "Spatial attention (particle head, RAW)",
    plots_dir / "attn_spatial_simple.png",
)
mlflow.log_figure(fig, "graficos/attn_spatial_simple.png")

```

### Attention versus x_center



```python
def plot_attention_vs_xcenter(
    attn_scores: np.ndarray,
    sequences: np.ndarray,
    valid_lengths: np.ndarray,
    title: str,
    save_path: Path,
    bins: int = 15,
    max_len: int = 120,
) -> None:
    """
    Plot mean received attention as a function of detector x_center position.

    Bins all hit-level (attention, x_center) pairs into spatial bins along the x-axis and plots the mean ± 1σ attention per bin. This reveals whether the model exhibits a systematic east-west attention asymmetry across the detector array, which could indicate sensitivity to shower azimuth or array geometry artifacts.

    Parameters
    ----------
    attn_scores : np.ndarray
        Attention score array of shape (batch, heads, L, L) or (batch, L, L). If 4D, the head dimension is averaged before processing.
    sequences : np.ndarray
        Padded sequence array of shape (n_events, seq_len, n_features). Column index 4 must contain x_center values in meters.
    valid_lengths : np.ndarray
        Integer array of shape (batch,) with unpadded sequence lengths.
    title : str
        Plot title, also used in the warning log if no valid data is found.
    save_path : Path
        Destination path for saving the figure at 200 dpi.
    bins : int, optional
        Number of spatial bins along the x_center axis. Default is 15.
    max_len : int, optional
        Maximum sequence length to process per event. Default is 120.

    Returns
    -------
    None
        Figure is saved to save_path and rendered via plt.show(). Logs a warning and returns early if no valid data is found.

    Notes
    -----
    The standard deviation per bin uses ddof=0 (population) because each bin aggregates contributions from multiple events, not a sample from a larger population.
    """
    if attn_scores.ndim == 4:
        attn_scores = attn_scores.mean(axis=1)
    xs, attn_vals = [], []
    for mat, seq, L in zip(attn_scores, sequences, valid_lengths):
        L = int(min(L, mat.shape[0], seq.shape[0], max_len))
        if L <= 0:
            continue
        attn_recv = mat[:L, :L].mean(axis=0)  # atención recibida por key-pos
        x_centers = seq[:L, 4]  # idx 4 = x_center
        mask = np.isfinite(attn_recv) & np.isfinite(x_centers)
        xs.append(x_centers[mask])
        attn_vals.append(attn_recv[mask])
    if not xs:
        logging.warning(
            "No hay datos válidos para graficar atención vs x_center en %s", title
        )
        return
    xs = np.concatenate(xs)
    attn_vals = np.concatenate(attn_vals)

    # Bineo y promedios
    bin_edges = np.linspace(xs.min(), xs.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    digitized = np.digitize(xs, bin_edges[:-1], right=False)
    mean_attn, std_attn, counts = [], [], []
    for b in range(1, bins + 1):
        m = digitized == b
        if m.any():
            mean_attn.append(attn_vals[m].mean())
            std_attn.append(attn_vals[m].std(ddof=0))
            counts.append(m.sum())
        else:
            mean_attn.append(np.nan)
            std_attn.append(np.nan)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bin_centers, mean_attn, color="tab:blue", label="Mean attention")
    ax.fill_between(
        bin_centers,
        np.array(mean_attn) - np.array(std_attn),
        np.array(mean_attn) + np.array(std_attn),
        color="tab:blue",
        alpha=0.2,
        label="±1σ",
    )
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("Atención recibida")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    logging.info("✓ Guardado: %s", save_path)


# Ejecutar para head de partícula (RAW)

# --- Llamada: head de clasificación gamma-hadrón (RAW) ---
plot_attention_vs_xcenter(
    attn_class_raw_scores,
    X_sample,
    valid_lengths_raw,
    "Attention vs x_center (Particle head, RAW)",
    plots_dir / "attn_vs_xcenter_particle.png",
    bins=15,
    max_len=120,
)
mlflow.log_figure(fig, "graficos/attn_vs_xcenter_particle.png")

```

### Explainability Findings (Quantitative)

- Spatial: central vs peripheral detector attention.
- Temporal: entropy, self vs context attention.
- Correlations with energy/particle count.
- Embedding separability (variance explained).


---
This document summarizes the key metrics used to evaluate model performance in the three main tasks: Classification (Gamma/Hadron), Angular Reconstruction, and Energy Estimation.
---

## 1. Metrics Summary Table

<div align="center">

| Category              | Metric                    | Notation / Formula                                          | Interpretation                           | Target Value (Benchmark) |
| :-------------------- | :------------------------ | :---------------------------------------------------------- | :--------------------------------------- | :----------------------- |
| **🎯 Classification** | **Accuracy**              | $\frac{TP + TN}{Total}$                                     | % of correct predictions (overall)       | $85\% - 95\%$            |
| (Gamma/Hadron)        | **Precision**             | $\frac{TP}{TP + FP}$                                        | Purity of gamma sample                   | $> 90\%$                 |
|                       | **Recall (Efficiency)**   | $\epsilon_\gamma = \frac{TP}{TP + FN}$                      | Gamma retention efficiency               | $> 80\%$                 |
|                       | **F1-Score**              | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic balance                         | $> 85\%$                 |
|                       | **AUC-ROC**               | $\int TPR \, d(FPR)$                                        | Global discriminative ability            | $0.90 - 0.98$            |
|                       | **Q-Factor**              | $Q = \frac{\epsilon_\gamma}{\sqrt{\epsilon_h}}$             | **Figure of Merit (IACTs)**              | $3 - 7$                  |
| **📐 Angle**          | **MAE**                   | $\langle \lvert \hat{\theta} - \theta \rvert \rangle$       | Mean absolute error                      | $1.0^\circ - 3.0^\circ$  |
| (Direction)           | **PSF68** ($\sigma_{68}$) | $P_{68}(\lvert \hat{\theta} - \theta \rvert)$               | **Angular Resolution (68% Containment)** | $0.5^\circ - 1.5^\circ$  |
|                       | **Bias (Zenith)**         | $\langle \hat{\theta} - \theta \rangle$                     | Systematic zenith bias                   | $\pm 0.1^\circ$          |
|                       | **Opening Angle**         | $\psi = \arccos(\hat{\vec{v}} \cdot \vec{v})$               | Solid angle between 3D vectors           | Rayleigh distribution    |
| **⚡ Energy**         | **MAE**                   | $\langle \lvert \hat{E} - E \rvert \rangle$                 | Mean error (GeV)                         | Depends on $E$           |
| (Calorimetry)         | **Energy Bias**           | $\langle \frac{\hat{E} - E}{E} \rangle$                     | **Systematic Relative Bias**             | $<\pm 5\%$               |
|                       | **Energy Resolution**     | $\sigma_{68}(\frac{\hat{E} - E}{E})$                        | **Relative Resolution**                  | $15\% - 30\%$            |
|                       | **Migration**             | $P(\hat{E}_j \lvert E_i)$                                   | Energy confusion matrix                  | Dominant diagonal        |

</div>

---

## 🔍 Detailed Explanation of Key Metrics

### 1. Quality Factor ($Q$-Factor)

This is the fundamental metric for optimizing selection cuts in gamma-ray astronomy (as in MAGIC, CTA, or HAWC). It maximizes the signal ($\gamma$) while suppressing the dominant background (hadrons).

$$Q = \frac{\epsilon_\gamma}{\sqrt{\epsilon_h}}$$

Where:

- $\epsilon_\gamma$: Signal efficiency (Gamma Recall).
- $\epsilon_h$: Residual background efficiency (Hadron False Positive Rate).
- $\sqrt{\epsilon_h}$: Represents the statistical fluctuation of the background (Poisson).

**Interpretation:**

- A factor $Q=5$ means the instrument's sensitivity improves by a factor of 5 compared to no cuts, reducing the required observation time by a factor of 25 ($t \propto 1/Q^2$).

---

### 2. Li & Ma Significance ($\sigma_{LiMa}$)

This is the statistical standard for claiming the discovery of an astrophysical source. It compares events in the source region ($N_{on}$) with a control region ($N_{off}$).

$$S_{LiMa} = \sqrt{2 \left( N_{on} \ln \left[ \frac{1+\alpha}{\alpha} \left( \frac{N_{on}}{N_{on} + N_{off}} \right) \right] + N_{off} \ln \left[ (1+\alpha) \left( \frac{N_{off}}{N_{on} + N_{off}} \right) \right] \right)}$$

- $\alpha$: Exposure ratio between ON and OFF regions ($t_{on}/t_{off}$).
- **Discovery threshold:** $S > 5\sigma$ (random fluctuation probability $< 2.8 \times 10^{-7}$).

---

### 3. Angular Resolution (PSF - Point Spread Function)

Instead of using a Gaussian standard deviation ($\sigma$), cosmic ray experiments use the **68% Containment Radius** (PSF68), since error distributions often have long non-Gaussian tails.

- **Definition:** The angular value $\Psi$ such that 68% of reconstructed events have an error less than $\Psi$.
- **Importance:** Defines the telescope's ability to resolve extended source morphology and reduces background contamination for point sources.

---

### 4. Energy Metrics: Bias and Resolution

Since the cosmic ray spectrum follows a power law ($dN/dE \propto E^{-\gamma}$), small systematic errors can severely distort the measured flux.

#### **Energy Bias**

$$\text{Bias} = \left\langle \frac{E_{pred} - E_{true}}{E_{true}} \right\rangle$$

- **Goal:** $\approx 0\%$.
- **Risk:** A positive bias ($+10\%$) shifts low-energy events (very abundant) to higher energy bins, creating false signals or "harder spectra".

#### **Energy Resolution**

$$\frac{\sigma_E}{E} = \frac{1}{2} (\text{Percentile}_{84} - \text{Percentile}_{16}) \quad \text{of the distribution } \frac{\Delta E}{E}$$

- The robust interquartile range is used instead of the simple standard deviation to mitigate the effect of outliers.

---

### 5. Migration Matrix

Visualizes the probability that an event with true energy $E_i$ is reconstructed with energy $E_j$.

**Theoretical Example:**

| $E_{true}$ \ $E_{rec}$ | **300 GeV** | **500 GeV** | **800 GeV** |
| :--------------------: | :---------: | :---------: | :---------: |
|      **300 GeV**       |  **0.85**   |    0.12     |    0.03     |
|      **500 GeV**       |    0.10     |  **0.80**   |    0.10     |
|      **800 GeV**       |    0.05     |    0.15     |  **0.80**   |

- **Diagonal:** Represents reconstruction accuracy.
- **Upper Triangle:** "Spill-over" events (reconstructed with more energy than true).
- **Use:** This matrix is essential for the _Unfolding_ (deconvolution) process of the final energy spectrum.



