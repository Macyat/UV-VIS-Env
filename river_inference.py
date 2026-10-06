"""Water-quality parameter inference for river-water samples (ARFF spectra -> JSON).

Usage:
    python river_inference.py -file_name <xxx.arff> -re_train False [--variant pca|range]

Two historical pipelines are supported and selected with --variant:
  pca   (default) : the original pipeline; per-spectrum smoothing followed by
                    global standardization and a PCA fitted on the fly (3 comps),
                    then one pretrained model per parameter.
  range           : the Linux pipeline; WaterQualityModel wrappers for models/ and
                    plain pickles for models/wider_range, with a turbidity-based
                    branch (mean TUR <= 20) and AN averaged over all matching files.

Both variants were previously two separate files (Regression.py and
Regression_linux.py) differing only in path separators plus these two pipelines;
they are merged here so that the paths are built with os.path.join and the script
runs unchanged on Windows and Linux.

Known legacy issues (behaviour intentionally preserved, please cross-check with
on-site parameters before changing):
    1. the pca variant applies SNV as "global standardization over the whole
       matrix", inconsistent with the per-spectrum SNV used elsewhere;
    2. the pca variant fits PCA at inference time, so the result for a given
       spectrum depends on the other samples in the same batch;
    3. in the range variant `np.max(x_smooth > 1)` is evaluated as "any value > 1"
       because of operator precedence - kept as-is to avoid behaviour changes.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings
from types import ModuleType

import joblib
import numpy as np
import pywt
from scipy.io import arff
from scipy.signal import savgol_filter, wiener
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Per-parameter inference config for the pca variant:
#   (output field, model file, input feature, detection limit, range limit)
PCA_PARAM_CONFIGS = [
    ("TN", "TN_model.pkl", "SNV", 0.5, 4),
    ("COD", "COD_model.pkl", "SNV", 2, 20),
    ("KMNO", "KMNO_model.pkl", "SNV", 0.5, 8),
    ("TUR", "TUR_model.pkl", "SMOOTH", 0.5, 70),
    ("TP", "TP_model.pkl", "SNV", 0.02, 1.2),
    ("AN", "AN_model.pkl", "SNV", 0.025, 1.5),
]
OUTPUT_KEYS = ["TN", "COD", "KMNO", "TUR", "TP", "AN"]
TRUE_VALUES = ["True", "Yes", "Y", "1", "true", "y", "yes"]

# Guard limits for the range variant: (detection limit, range limit)
RANGE_GUARDS = {
    "KMNO": (0.5, 20),
    "TN": (0.2, 8),
    "TUR": (0.5, 150),
    "TP": (0.005, 5),
    "AN": (0.015, 5),
}
TURBIDITY_BRANCH_THRESHOLD = 20


def default_serializer(obj):
    if isinstance(obj, np.ndarray):
        if obj.size > 0:  # check whether the array is empty
            return obj.tolist()[0]
        else:
            return None  # or return another default value
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def standardize_data(data):
    """
    To implement snv on data
    :param data: the spectrum matrix
    :return: the spectrum matrix after snv processing
    """
    mean_val = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean_val) / std_dev
    return standardized_data


def wavelet_denoising(data, wavelet, level):
    coeff = pywt.wavedec(data, wavelet, mode="smooth", level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="smooth")


def smooth_rows(X1):
    """Wiener -> Savitzky-Golay(15, 3) -> sym4 denoising, row by row."""
    X1_smooth = X1.copy()
    for i in range(len(X1)):
        X1_smooth[i, :] = wiener(X1[i, :], mysize=None, noise=None)
        X1_smooth[i, :] = savgol_filter(X1_smooth[i, :], 15, 3)
        X1_smooth[i, :] = wavelet_denoising(X1_smooth[i, :], "sym4", 2)[1:]
    return X1_smooth


def process_range(X1):
    """Features for the range variant: per-spectrum SNV plus the smoothed matrix."""
    X1_smooth = smooth_rows(X1)
    X1_snv = X1.copy()
    for i in range(len(X1)):
        X1_snv[i, :] = standardize_data(X1_smooth[i, :])
    return X1_snv, X1_smooth


def process_pca(X1):
    """Features for the pca variant: global standardization + PCA fitted on the fly."""
    X1_smooth = smooth_rows(X1)
    X1_snv = standardize_data(X1_smooth)
    X1_snv = PCA(n_components=3).fit_transform(X1_snv)
    X1_smooth_pca = PCA(n_components=3).fit_transform(X1_smooth)
    return X1_snv, X1_smooth_pca


def check_positive_limit(value, name):
    """Validate that a limit value (detection limit / range limit) is positive"""
    if value <= 0:
        raise ValueError(f"{name}必须大于 0，当前为 {value}")


def detection_limit_guard(res, detection_limit):
    """Detection-limit guard: value at/below the limit is replaced by LOD/2."""
    check_positive_limit(detection_limit, "检出限")
    res1 = res.copy()
    for i in range(len(res)):
        if res[i] <= detection_limit:
            res1[i] = detection_limit / 2
    return res1


def range_limit_guard(res, range_limit):
    """Range guard: value above the range limit is capped at that limit."""
    check_positive_limit(range_limit, "量程上限")
    res1 = res.copy()
    for i in range(len(res)):
        if res[i] > range_limit:
            res1[i] = range_limit
    return res1


class MeanCenterer(BaseEstimator, TransformerMixin):
    """Custom mean-centering transformer"""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_

    def inverse_transform(self, X):
        return X + self.mean_


class YAutoScaler(BaseEstimator, TransformerMixin):
    """y auto-scaling transformer"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, y):
        # ensure y is a 2D array (n_samples, n_targets)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.std(y, axis=0)
        # avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        return (y - self.mean_) / self.scale_

    def inverse_transform(self, y_scaled):
        return y_scaled * self.scale_ + self.mean_


class AutoScaler_dummy(BaseEstimator, TransformerMixin):
    """y auto-scaling transformer"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, y):
        return self

    def transform(self, y):
        return y

    def inverse_transform(self, y_scaled):
        return y_scaled


class WaterQualityModel:
    """Model wrapper dedicated to water-quality monitoring"""

    def __init__(self, model):
        self.x_scaler = MeanCenterer()
        self.y_scaler = YAutoScaler()
        self.model = model

    def fit(self, X, y):
        """Train the model (including preprocessing)"""
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        """Predict water-quality parameters (including inverse transform)"""
        X_scaled = self.x_scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_pred_scaled)

    def save(self, filename):
        """Save the complete model"""
        joblib.dump({"x_scaler": self.x_scaler, "y_scaler": self.y_scaler, "model": self.model}, filename)

    @staticmethod
    def load(filename):
        """Load the model"""
        return joblib.load(filename)


def load_model(path, joblib_format):
    """Load a model file.

    Some artifacts were pickled against a ``models`` namespace, so a stub module is
    registered while unpickling and the previous entry is restored afterwards.
    """
    previous = sys.modules.get("models")
    stub = ModuleType("models")
    for cls in (MeanCenterer, YAutoScaler, AutoScaler_dummy, WaterQualityModel):
        setattr(stub, cls.__name__, cls)
    sys.modules["models"] = stub
    try:
        if joblib_format:
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        if previous is None:
            sys.modules.pop("models", None)
        else:
            sys.modules["models"] = previous


def first_match(directory, prefix):
    """Return the first model file matching ``prefix*`` inside ``directory``."""
    matches = sorted(glob.glob(os.path.join(directory, prefix + "*")))
    if not matches:
        raise FileNotFoundError(f"no model matching {prefix}* in {directory}")
    return matches[0]


def parse_test_id(file_name):
    """Parse the test id out of the file name: xxx_<id>.arff"""
    parts = os.path.splitext(file_name)[0].split("_")
    if len(parts) < 2:
        raise ValueError(f"文件名不符合 xxx_<id>.arff 格式：{file_name}")
    return parts[1]


def load_spectra(file_path):
    """Load the ARFF file and return (sample id list, spectra matrix)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"光谱文件不存在：{file_path}")
    data, _ = arff.loadarff(file_path)
    ids = [int(i[0]) for i in data]
    spectra = np.array([list(i)[1:-1] for i in data])
    return ids, spectra


def predict_pca(models_dir, features):
    """Load each parameter model and apply both guards; return {field: predictions}."""
    results = {}
    for field, model_file, feature_name, detection_limit, range_limit in PCA_PARAM_CONFIGS:
        check_positive_limit(detection_limit, "检出限")
        check_positive_limit(range_limit, "量程上限")
        if range_limit <= detection_limit:
            raise ValueError(
                f"{field} 量程上限({range_limit})必须大于检出限({detection_limit})"
            )
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件缺失：{model_path}")

        model = load_model(model_path, joblib_format=False)
        X = features[feature_name]
        res = model.predict(X)
        if len(res) != len(X):
            raise ValueError(f"{field} 预测数({len(res)})与样本数({len(X)})不一致")
        res = detection_limit_guard(res, detection_limit)
        results[field] = range_limit_guard(res, range_limit)
    return results


def predict_range(models_dir, x_smooth):
    """Port of the Linux pipeline: wider_range pickles vs joblib wrappers in models/."""
    wider_dir = os.path.join(models_dir, "wider_range")

    def guarded(field, res):
        detection_limit, range_limit = RANGE_GUARDS[field]
        return range_limit_guard(detection_limit_guard(res, detection_limit), range_limit)

    def load_from(directory, prefix, joblib_format):
        path = first_match(directory, prefix)
        return load_model(path, joblib_format=joblib_format)

    # initial turbidity always comes from wider_range and decides which branch to use
    results = {"TUR": guarded("TUR", load_from(wider_dir, "TUR", False).predict(x_smooth))}
    results["COD"] = [None] * len(x_smooth)

    if np.mean(results["TUR"]) <= TURBIDITY_BRANCH_THRESHOLD:
        for field in ("KMNO", "TN", "TUR", "TP"):
            results[field] = guarded(field, load_from(models_dir, field, True).predict(x_smooth))
        # AN is averaged over every matching model file
        an_predictions = [
            load_model(path, joblib_format=True).predict(x_smooth)
            for path in sorted(glob.glob(os.path.join(models_dir, "AN*")))
        ]
        results["AN"] = guarded("AN", np.mean(an_predictions, axis=0))
    else:
        for field in ("KMNO", "TN", "TP", "AN"):
            results[field] = guarded(field, load_from(wider_dir, field, False).predict(x_smooth))

    # legacy precedence quirk kept verbatim: this is "any value above 1"
    if np.max(x_smooth > 1):
        results["KMNO"] = guarded(
            "KMNO", load_from(wider_dir, "KMNO", False).predict(x_smooth)
        )

    results["TN"] = np.maximum(results["TN"], results["AN"])
    return results


def build_items(ids, results):
    """Assemble output items; field order is fixed as id, TN, COD, KMNO, TUR, TP, AN"""
    return [
        dict({"id": id_i}, **{field: results[field][i] for field in OUTPUT_KEYS})
        for i, id_i in enumerate(ids)
    ]


def run(file_path, retrain, variant):
    file_dir = os.path.dirname(file_path) or "."
    file_name = os.path.basename(file_path)
    test_id = parse_test_id(file_name)

    ids, spectra = load_spectra(os.path.join(file_dir, file_name))
    spectra = spectra[:, 10:-10]

    if retrain:
        raise ValueError("重训分支尚未实现：本脚本仅支持预训练模型推理，请传 -re_train False")

    models_dir = os.path.join(file_dir, "models")
    if variant == "pca":
        x_snv, x_smooth = process_pca(spectra)
        results = predict_pca(models_dir, {"SNV": x_snv, "SMOOTH": x_smooth})
    elif variant == "range":
        _, x_smooth = process_range(spectra)
        results = predict_range(models_dir, x_smooth)
    else:  # argparse already restricts the choices, kept as a safety net
        raise ValueError(f"unknown variant: {variant}")

    res_dict = {
        "id": test_id,
        "item": build_items(ids, results),
        "time": int(time.time() * 1000),
    }
    out_path = os.path.join(file_dir, f"result_{test_id}.json")
    with open(out_path, "w") as f:
        json.dump(res_dict, f, default=default_serializer, indent=4)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Predicting water quality with UV-VIS spectrum with pretrained models"
    )
    parser.add_argument("-file_name", type=str, help="File to open")
    parser.add_argument("-re_train", type=str, help="Retrain or not")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["pca", "range"],
        default="pca",
        help="inference pipeline: pca (original) or range (legacy Linux, wider_range)",
    )
    args = parser.parse_args()

    out_path = run(args.file_name, args.re_train in TRUE_VALUES, args.variant)
    print(f"结果已写入：{out_path}")


if __name__ == "__main__":
    main()
