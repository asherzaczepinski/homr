#!/usr/bin/env python3
"""
Script to update imports in all reorganized homr files.
"""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Core imports
    r'from homr\.model import': 'from homr.core.model import',
    r'from homr\.bounding_boxes import': 'from homr.core.bounding_boxes import',
    r'from homr\.type_definitions import': 'from homr.core.type_definitions import',
    r'from homr\.constants import': 'from homr.core.constants import',
    r'from homr import constants': 'from homr.core import constants',

    # Preprocessing imports
    r'from homr\.deskew import': 'from homr.preprocessing.deskew import',
    r'from homr\.resize import': 'from homr.preprocessing.resize import',
    r'from homr\.color_adjust import': 'from homr.preprocessing.color_adjust import',
    r'from homr\.autocrop import': 'from homr.preprocessing.autocrop import',
    r'from homr\.noise_filtering import': 'from homr.preprocessing.noise_filtering import',
    r'from homr\.image_utils import': 'from homr.preprocessing.image_utils import',
    r'from homr import color_adjust': 'from homr import preprocessing',

    # Detection imports
    r'from homr\.staff_detection import': 'from homr.detection.staff_detection import',
    r'from homr\.note_detection import': 'from homr.detection.note_detection import',
    r'from homr\.bar_line_detection import': 'from homr.detection.bar_line_detection import',
    r'from homr\.brace_dot_detection import': 'from homr.detection.brace_dot_detection import',
    r'from homr\.title_detection import': 'from homr.detection.title_detection import',

    # Processing imports
    r'from homr\.staff_parsing import': 'from homr.processing.staff_parsing import',
    r'from homr\.staff_dewarping import': 'from homr.processing.staff_dewarping import',
    r'from homr\.staff_regions import': 'from homr.processing.staff_regions import',
    r'from homr\.staff_position_save_load import': 'from homr.processing.staff_position_save_load import',
    r'from homr\.staff_parsing_tromr import': 'from homr.processing.staff_parsing_tromr import',

    # Output imports
    r'from homr\.music_xml_generator import': 'from homr.output.music_xml_generator import',
    r'from homr\.visualization_output import': 'from homr.output.visualization_output import',

    # Utils imports
    r'from homr\.debug import': 'from homr.utils.debug import',
    r'from homr\.simple_logging import': 'from homr.utils.simple_logging import',
    r'from homr\.download_utils import': 'from homr.utils.download_utils import',
    r'from homr\.circle_of_fifths import': 'from homr.utils.circle_of_fifths import',
    r'from homr import download_utils': 'from homr.utils import download_utils',
}

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply all import mappings
        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            content = re.sub(old_pattern, new_import, content)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    homr_dir = Path(__file__).parent / 'homr'

    # Find all Python files to update
    directories_to_update = [
        'core',
        'preprocessing',
        'detection',
        'processing',
        'output',
        'utils',
        'segmentation',
        'transformer',
    ]

    updated_files = []

    for directory in directories_to_update:
        dir_path = homr_dir / directory
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue  # Skip __init__.py files

            if update_imports_in_file(py_file):
                updated_files.append(py_file)
                print(f"✓ Updated: {py_file.relative_to(homr_dir)}")

    print(f"\nTotal files updated: {len(updated_files)}")

if __name__ == '__main__':
    main()
