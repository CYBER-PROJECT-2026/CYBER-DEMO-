"""
aegis_ml/label_encoding.py
==========================
Independent label encoder for the AEGIS-IoT retraining and evaluation pipeline.

PURPOSE:
- Maps string attack class labels to integer IDs for model training:
    'DDoS'     -> 0
    'DoS'      -> 1
    'Spoofing' -> 2
- Maps integer IDs back to string attack labels for inference and reporting.
- Strictly validates input labels and IDs, raising clear ValueErrors on any
  unexpected or unknown values.
- Completely independent: does NOT load, inherit from, or reference the
  existing project label_encoder.pkl artifact.
"""

from pathlib import Path
from typing import Sequence, Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from aegis_ml.feature_schema import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
)


class AegisLabelEncoder:
    """
    Independent label encoder implementing the canonical AEGIS-IoT 3-class mapping:
        DDoS     -> 0
        DoS      -> 1
        Spoofing -> 2
    """

    def __init__(self):
        # Explicit canonical class definitions
        self.classes: List[str] = list(CLASS_NAMES)
        self.class_to_idx: Dict[str, int] = dict(CLASS_TO_INDEX)
        self.idx_to_class: Dict[int, str] = dict(INDEX_TO_CLASS)
        self.num_classes: int = len(self.classes)
        self.is_fitted: bool = True  # Canonical mapping is predefined

    @property
    def classes_(self) -> np.ndarray:
        """Alias for compatibility with scikit-learn LabelEncoder interface."""
        return np.array(self.classes)

    def fit(self, y: Optional[Sequence[str]] = None) -> "AegisLabelEncoder":
        """
        Validates that supplied labels are compatible with the canonical classes.
        Does not alter the predefined canonical mapping.
        """
        if y is not None:
            unique_labels = set(pd.Series(y).dropna().unique())
            unexpected = unique_labels - set(self.classes)
            if unexpected:
                raise ValueError(
                    f"Label fitting failed: Found unexpected class labels {unexpected}. "
                    f"Allowed canonical classes are: {self.classes}"
                )
        return self

    def transform(self, y: Sequence[str]) -> np.ndarray:
        """
        Encodes a sequence of string class labels into canonical integer IDs.

        Parameters:
            y: Sequence of string class labels (e.g. ['DDoS', 'DoS', 'Spoofing']).

        Returns:
            np.ndarray: Array of integer IDs (dtype=np.int64).

        Raises:
            ValueError: If any label is unknown or NaN.
        """
        if isinstance(y, (pd.Series, pd.Index)):
            labels_list = y.tolist()
        else:
            labels_list = list(y)

        encoded = []
        for idx, label in enumerate(labels_list):
            if pd.isna(label):
                raise ValueError(f"Label encoding failed: NaN value found at index {idx}.")
            label_str = str(label).strip()
            if label_str not in self.class_to_idx:
                raise ValueError(
                    f"Label encoding failed: Unknown label '{label_str}' at index {idx}. "
                    f"Allowed classes are strictly: {self.classes}"
                )
            encoded.append(self.class_to_idx[label_str])

        return np.array(encoded, dtype=np.int64)

    def fit_transform(self, y: Sequence[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y_indices: Sequence[int]) -> np.ndarray:
        """
        Decodes a sequence of integer IDs back into canonical string labels.

        Parameters:
            y_indices: Sequence of integer class IDs (0, 1, or 2).

        Returns:
            np.ndarray: Array of string class labels.

        Raises:
            ValueError: If any integer ID is invalid.
        """
        decoded = []
        for idx, val in enumerate(y_indices):
            try:
                int_val = int(val)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Inverse transform failed: Non-integer class ID '{val}' at index {idx}."
                ) from e

            if int_val not in self.idx_to_class:
                raise ValueError(
                    f"Inverse transform failed: Invalid class ID {int_val} at index {idx}. "
                    f"Valid class IDs are: {list(self.idx_to_class.keys())}"
                )
            decoded.append(self.idx_to_class[int_val])

        return np.array(decoded, dtype=object)

    def save(self, destination_path: Union[str, Path]) -> Path:
        """
        Saves this encoder object using joblib.
        Ensures destination directory exists.
        """
        dest = Path(destination_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "AegisLabelEncoder":
        """
        Loads a saved AegisLabelEncoder from disk and validates its integrity.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Label encoder artifact not found at: {path}")

        encoder = joblib.load(path)
        if not isinstance(encoder, cls):
            raise TypeError(
                f"Loaded artifact is not an {cls.__name__}. Got type: {type(encoder)}"
            )

        # Validate internal mapping
        if encoder.class_to_idx != CLASS_TO_INDEX:
            raise ValueError(
                f"Loaded encoder has invalid class mapping: {encoder.class_to_idx}"
            )

        return encoder


if __name__ == "__main__":
    print("Testing aegis_ml.label_encoding...")
    encoder = AegisLabelEncoder()
    sample_labels = ["DDoS", "DoS", "Spoofing", "DoS", "DDoS"]
    encoded = encoder.transform(sample_labels)
    decoded = encoder.inverse_transform(encoded)
    
    assert list(encoded) == [0, 1, 2, 1, 0], f"Encoding mismatch: {encoded}"
    assert list(decoded) == sample_labels, f"Decoding mismatch: {decoded}"
    print(f"Self-check passed: {sample_labels} -> {encoded} -> {decoded}")
