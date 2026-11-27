"""Traing code"""
import logging

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.base import BaseEstimator


def quiet_logger():
    logging.getLogger("mlflow").setLevel(logging.ERROR)


def export_onnx(model: BaseEstimator, float_cols: int = 1, string_cols: int|None = None):
    """Convert the fitted regressor to an ONNX model with input shape [None, 1]:
    one scalar float32 input per row, one prediction per row."""
    
    initial_types=[
        ("numerical", FloatTensorType([None, float_cols]))
    ]
    if string_cols:
        initial_types = [
            ("categorical", StringTensorType([None, string_cols]))
        ] + initial_types
         
    
    return to_onnx(
        model,
        initial_types=initial_types,  # ty: ignore[invalid-argument-type]
        target_opset=21,
    )

