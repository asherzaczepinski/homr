"""
Detection modules for musical symbols in sheet music.

This package contains all detection algorithms:
- Staff line detection
- Note head detection
- Accidental detection (sharps, flats, naturals)
- Bar line detection
- Brace/bracket detection
- Title/text detection
"""

from homr.detection.staff_detection import (
    detect_staff,
    break_wide_fragments,
    make_lines_stronger,
)
from homr.detection.note_detection import (
    add_notes_to_staffs,
    combine_noteheads_with_stems,
)
from homr.detection.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.detection.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.detection.title_detection import detect_title, download_ocr_weights

__all__ = [
    "detect_staff",
    "break_wide_fragments",
    "make_lines_stronger",
    "add_notes_to_staffs",
    "combine_noteheads_with_stems",
    "detect_bar_lines",
    "prepare_bar_line_image",
    "find_braces_brackets_and_grand_staff_lines",
    "prepare_brace_dot_image",
    "detect_title",
    "download_ocr_weights",
]
