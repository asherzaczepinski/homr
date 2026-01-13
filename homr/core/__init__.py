"""
Core data models and utilities for the homr OMR system.

This package contains fundamental data structures used throughout the codebase:
- InputPredictions, Staff, MultiStaff models
- Bounding box classes and operations
- Type definitions and constants
"""

from homr.core.bounding_boxes import (
    BoundingBox,
    BoundingEllipse,
    RotatedBoundingBox,
    AngledBoundingBox,
    DebugDrawable,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.core.model import (
    InputPredictions,
    Staff,
    MultiStaff,
    Note,
    StemDirection,
    SymbolOnStaff,
)
from homr.core.type_definitions import NDArray
from homr.core import constants

__all__ = [
    "BoundingBox",
    "BoundingEllipse",
    "RotatedBoundingBox",
    "AngledBoundingBox",
    "DebugDrawable",
    "create_bounding_ellipses",
    "create_rotated_bounding_boxes",
    "InputPredictions",
    "Staff",
    "MultiStaff",
    "Note",
    "StemDirection",
    "SymbolOnStaff",
    "NDArray",
    "constants",
]
