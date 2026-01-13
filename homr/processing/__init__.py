"""
Staff processing and parsing modules.

This package handles:
- Staff parsing and symbol recognition
- Staff dewarping/straightening
- Staff region management
- Staff position persistence
"""

from homr.processing.staff_parsing import parse_staffs
from homr.processing.staff_dewarping import dewarp_staff_image, StaffDewarping
from homr.processing.staff_regions import StaffRegions
from homr.processing.staff_position_save_load import (
    save_staff_positions,
    load_staff_positions,
)

__all__ = [
    "parse_staffs",
    "dewarp_staff_image",
    "StaffDewarping",
    "StaffRegions",
    "save_staff_positions",
    "load_staff_positions",
]
