"""
Accidental detection using YOLOv10 model trained on DeepScores.

This module was extracted from Orchestra-AI-2 and provides high-accuracy
detection of musical accidentals (sharps, flats, naturals, double sharps, etc.)

Model: YOLOv10 trained on DeepScores dataset
Location: homr/models/accidentals/best.pt
"""

from homr.detection.accidentals.accidental_detector import (
    AccidentalDetector,
    ACCIDENTAL_CLASSES,
)

__all__ = [
    "AccidentalDetector",
    "ACCIDENTAL_CLASSES",
]
