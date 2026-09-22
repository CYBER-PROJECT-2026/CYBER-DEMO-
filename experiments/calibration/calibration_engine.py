"""
Calibration Engine for Multi-Class IoT Intrusion Detection.
Provides modular implementations of:
1. Multiclass Sigmoid / Platt Scaling Calibrator
2. Multiclass Isotonic Regression Calibrator
3. Multiclass Temperature Scaling Calibrator
4. Comprehensive Calibration & Classification Metrics (ECE, MCE, Classwise ECE, Brier, Log Loss)
5. Reliability Diagram Plotting
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import softmax
import matplotlib.pyplot as plt


class MulticlassSigmoidCalibrator:
    """
    Multiclass Platt / Sigmoid Calibrator.
    Fits 1-vs-Rest Platt scaling logistic models per class on raw logits / decision function values,
    and applies vector L1 normalization to guarantee valid multiclass probabilities (sum = 1, p >= 0).
    """
    def __init__(self):
        self.params = []  # List of (a_k, b_k) for each class
        self.num_classes = None

    def fit(self, logits_or_probs, y_true, is_logits=True):
        """
        logits_or_probs: (N, K) array of raw model logits or probabilities
        y_true: (N,) array of true class labels (0 to K-1)
        """
        N, K = logits_or_probs.shape
        self.num_classes = K
        self.params = []

        if is_logits:
            logits = logits_or_probs
        else:
            eps = 1e-7
            probs_clipped = np.clip(logits_or_probs, eps, 1.0 - eps)
            logits = np.log(probs_clipped / (1.0 - probs_clipped))

        for k in range(K):
            y_binary = (y_true == k).astype(np.float64)
            z_k = logits[:, k]

            # Objective: Binary cross-entropy with respect to a_k * z_k + b_k
            def loss_fn(p):
                a, b = p
                s = np.clip(a * z_k + b, -50.0, 50.0)
                # Numerically stable log-loss
                log_loss = np.maximum(s, 0.0) - s * y_binary + np.log(1.0 + np.exp(-np.abs(s)))
                return np.mean(log_loss)

            res = minimize(loss_fn, x0=[1.0, 0.0], method='L-BFGS-B', bounds=[(0.01, 10.0), (-20.0, 20.0)])
            self.params.append((float(res.x[0]), float(res.x[1])))

        return self

    def predict_proba(self, logits_or_probs, is_logits=False):
        """
        Calibrate probabilities and normalize to sum to 1.
        """
        if is_logits:
            logits = logits_or_probs
        else:
            eps = 1e-7
            probs_clipped = np.clip(logits_or_probs, eps, 1.0 - eps)
            logits = np.log(probs_clipped / (1.0 - probs_clipped))

        N, K = logits.shape
        calib_probs = np.zeros((N, K), dtype=np.float64)

        for k in range(K):
            a, b = self.params[k]
            s = np.clip(a * logits[:, k] + b, -50.0, 50.0)
            calib_probs[:, k] = 1.0 / (1.0 + np.exp(-s))

        # Normalize across classes
        row_sums = np.sum(calib_probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calib_probs = calib_probs / row_sums
        return calib_probs


class MulticlassIsotonicCalibrator:
    """
    Multiclass Isotonic Regression Calibrator.
    Fits 1-vs-Rest Isotonic Regressors per class on raw predicted probabilities,
    and applies L1 normalization across classes to guarantee sum = 1, p >= 0.
    """
    def __init__(self):
        self.calibrators = []
        self.num_classes = None

    def fit(self, probs, y_true):
        N, K = probs.shape
        self.num_classes = K
        self.calibrators = []

        for k in range(K):
            y_binary = (y_true == k).astype(np.float64)
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(probs[:, k], y_binary)
            self.calibrators.append(iso)

        return self

    def predict_proba(self, probs):
        N, K = probs.shape
        calib_probs = np.zeros((N, K), dtype=np.float64)

        for k in range(K):
            calib_probs[:, k] = self.calibrators[k].predict(probs[:, k])

        # Ensure no negative values and normalize
        calib_probs = np.clip(calib_probs, 1e-12, 1.0)
        row_sums = np.sum(calib_probs, axis=1, keepdims=True)
        calib_probs = calib_probs / row_sums
        return calib_probs


class MulticlassTemperatureCalibrator:
    """
    Multiclass Temperature Scaling Calibrator (Guo et al., 2017).
    Learns a single scalar temperature T > 0 on pre-softmax logits z:
    P(y=k | x) = exp(z_k / T) / sum_j exp(z_j / T)
    Optimizes T by minimizing multiclass negative log likelihood (cross-entropy log loss).
    Strictly preserves the argmax class ranking (top-1 accuracy) while fixing confidence miscalibration.
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        """
        logits: (N, K) array of unnormalized logits
        y_true: (N,) array of true class labels (0 to K-1)
        """
        N, K = logits.shape

        def nll_loss(t):
            temp = t[0]
            scaled_logits = logits / temp
            # Stable log-softmax
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp

            # Negative log likelihood of true labels
            true_log_probs = log_probs[np.arange(N), y_true]
            return -np.mean(true_log_probs)

        res = minimize(nll_loss, x0=[1.0], method='L-BFGS-B', bounds=[(0.01, 50.0)])
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, logits):
        """
        Computes calibrated probabilities from logits using learned temperature.
        """
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=1)

    def calibrate_from_probs(self, probs):
        """
        Convenience method if only probabilities are available: converts to logits, then scales.
        """
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs_clipped)
        return self.predict_proba(logits)


# =====================================================================
# METRICS COMPUTATION
# =====================================================================

def compute_multiclass_log_loss(y_true, y_prob, eps=1e-15):
    """
    Multiclass Cross-Entropy / Log Loss.
    """
    N, K = y_prob.shape
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    # Normalize clipped probabilities
    y_prob_clipped = y_prob_clipped / np.sum(y_prob_clipped, axis=1, keepdims=True)
    
    # One-hot encode y_true
    y_one_hot = np.zeros((N, K), dtype=np.float64)
    y_one_hot[np.arange(N), y_true] = 1.0

    return float(-np.mean(np.sum(y_one_hot * np.log(y_prob_clipped), axis=1)))


def compute_multiclass_brier_score(y_true, y_prob):
    """
    Multiclass Brier Score: Mean squared error across all K classes.
    BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    N, K = y_prob.shape
    y_one_hot = np.zeros((N, K), dtype=np.float64)
    y_one_hot[np.arange(N), y_true] = 1.0

    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def compute_calibration_curve_and_ece(y_true, y_prob, n_bins=10):
    """
    Top-class (Confidence-based) Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    N = len(y_true)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count_in_bin = int(np.sum(in_bin))

        if count_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(accuracy_in_bin - confidence_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(confidence_in_bin)
            bin_counts.append(count_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "n_bins": n_bins
    }


def compute_classwise_ece(y_true, y_prob, n_bins=10):
    """
    Classwise (Marginal) Expected Calibration Error across all K classes.
    """
    N, K = y_prob.shape
    class_eces = []
    per_class_curves = {}

    for k in range(K):
        y_binary = (y_true == k).astype(np.float64)
        prob_k = y_prob[:, k]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        cls_ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (prob_k > bin_lower) & (prob_k <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            count_in_bin = int(np.sum(in_bin))

            if count_in_bin > 0:
                acc = float(np.mean(y_binary[in_bin]))
                conf = float(np.mean(prob_k[in_bin]))
                cls_ece += abs(acc - conf) * prop_in_bin
                bin_accs.append(acc)
                bin_confs.append(conf)
                bin_counts.append(count_in_bin)
            else:
                bin_accs.append(0.0)
                bin_confs.append((bin_lower + bin_upper) / 2.0)
                bin_counts.append(0)

        class_eces.append(float(cls_ece))
        per_class_curves[k] = {
            "ece": float(cls_ece),
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_counts": bin_counts
        }

    return {
        "classwise_ece": float(np.mean(class_eces)),
        "per_class_ece": class_eces,
        "per_class_curves": per_class_curves
    }


def generate_reliability_plots(results_dict, class_names, output_png_path):
    """
    Generate comprehensive publication-grade reliability diagrams comparing all methods.
    """
    methods = ["Original XGBoost", "Sigmoid / Platt", "Isotonic", "Temperature Scaling"]
    keys = ["original", "sigmoid", "isotonic", "temperature"]
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

    # 1. Top-Class Confidence Reliability Diagram
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7, lw=1.5)
    for m_key, m_name, col in zip(keys, methods, colors):
        curve = results_dict[m_key]["calibration_top"]["bin_confs"]
        accs = results_dict[m_key]["calibration_top"]["bin_accs"]
        counts = results_dict[m_key]["calibration_top"]["bin_counts"]
        # Filter non-empty bins for plotting
        valid = [i for i, c in enumerate(counts) if c > 0]
        x = [curve[i] for i in valid]
        y = [accs[i] for i in valid]
        ece_val = results_dict[m_key]["calibration_top"]["ece"]
        ax1.plot(x, y, marker='o', lw=2, color=col, label=f"{m_name} (ECE={ece_val*100:.2f}%)")

    ax1.set_title("Top-Class Reliability Diagram (Confidence vs Empirical Accuracy)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax1.set_ylabel("Empirical Accuracy", fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper left", frameon=True)

    # 2. Per-Class Calibration Curves for Best / All Methods (Class 0: DDoS)
    ax2 = axes[0, 1]
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7, lw=1.5)
    for m_key, m_name, col in zip(keys, methods, colors):
        c_curve = results_dict[m_key]["calibration_classwise"]["per_class_curves"][0]
        valid = [i for i, c in enumerate(c_curve["bin_counts"]) if c > 0]
        x = [c_curve["bin_confs"][i] for i in valid]
        y = [c_curve["bin_accs"][i] for i in valid]
        ax2.plot(x, y, marker='s', lw=1.8, color=col, label=f"{m_name} (ECE={c_curve['ece']*100:.2f}%)")

    ax2.set_title(f"Class Reliability: {class_names[0]} (DDoS)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted Probability", fontsize=10)
    ax2.set_ylabel("True Empirical Frequency", fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(loc="upper left", frameon=True)

    # 3. Class 1: DoS
    ax3 = axes[1, 0]
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7, lw=1.5)
    for m_key, m_name, col in zip(keys, methods, colors):
        c_curve = results_dict[m_key]["calibration_classwise"]["per_class_curves"][1]
        valid = [i for i, c in enumerate(c_curve["bin_counts"]) if c > 0]
        x = [c_curve["bin_confs"][i] for i in valid]
        y = [c_curve["bin_accs"][i] for i in valid]
        ax3.plot(x, y, marker='^', lw=1.8, color=col, label=f"{m_name} (ECE={c_curve['ece']*100:.2f}%)")

    ax3.set_title(f"Class Reliability: {class_names[1]} (DoS)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Predicted Probability", fontsize=10)
    ax3.set_ylabel("True Empirical Frequency", fontsize=10)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.legend(loc="upper left", frameon=True)

    # 4. Class 2: Spoofing
    ax4 = axes[1, 1]
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7, lw=1.5)
    for m_key, m_name, col in zip(keys, methods, colors):
        c_curve = results_dict[m_key]["calibration_classwise"]["per_class_curves"][2]
        valid = [i for i, c in enumerate(c_curve["bin_counts"]) if c > 0]
        x = [c_curve["bin_confs"][i] for i in valid]
        y = [c_curve["bin_accs"][i] for i in valid]
        ax4.plot(x, y, marker='d', lw=1.8, color=col, label=f"{m_name} (ECE={c_curve['ece']*100:.2f}%)")

    ax4.set_title(f"Class Reliability: {class_names[2]} (Spoofing)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Predicted Probability", fontsize=10)
    ax4.set_ylabel("True Empirical Frequency", fontsize=10)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"✓ Saved reliability diagram figure to: {output_png_path}")
