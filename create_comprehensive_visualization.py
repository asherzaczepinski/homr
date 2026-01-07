#!/usr/bin/env python3
"""
Create a comprehensive visualization showing all detected elements on the original image.
"""
import sys
import cv2
import numpy as np

from homr import color_adjust
from homr.autocrop import autocrop
from homr.bar_line_detection import detect_bar_lines, prepare_bar_line_image
from homr.bounding_boxes import create_bounding_ellipses, create_rotated_bounding_boxes
from homr.brace_dot_detection import find_braces_brackets_and_grand_staff_lines, prepare_brace_dot_image
from homr.debug import Debug
from homr.model import InputPredictions
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.inference_segnet import extract
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger


def create_comprehensive_visualization(image_path: str, output_path: str) -> None:
    """Create a single image with all detections annotated."""
    # Load and preprocess
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read {image_path}")

    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _ = color_adjust.color_adjust(image, 40)

    # Get predictions
    result = extract(preprocessed, image_path, step_size=320, use_cache=False, use_gpu_inference=False)
    original_image = cv2.resize(image, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))

    predictions = InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )

    debug = Debug(predictions.original, image_path, False)
    predictions = filter_predictions(predictions, debug)
    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

    # Create bounding boxes for all elements
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    staff_fragments = break_wide_fragments(staff_fragments)
    noteheads_with_stems = combine_noteheads_with_stems(noteheads, stems_rest)

    if len(noteheads_with_stems) > 0:
        average_note_head_height = float(
            np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        )
    else:
        average_note_head_height = 16.0

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)

    staffs = detect_staff(
        debug, predictions.staff, staff_fragments, clefs_keys, bar_line_boxes
    )

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

    # Create comprehensive visualization
    img = original_image.copy()

    # Define colors for different elements
    STAFF_COLOR = (0, 255, 0)        # Green for staffs
    NOTE_COLOR = (255, 0, 0)         # Blue for notes
    BAR_LINE_COLOR = (0, 0, 255)     # Red for bar lines
    CLEF_COLOR = (255, 255, 0)       # Cyan for clefs/keys

    # Draw staffs
    for staff in staffs:
        staff.draw_onto_image(img, STAFF_COLOR)

    # Draw bar lines
    for bar_line in bar_line_boxes:
        bar_line.draw_onto_image(img, BAR_LINE_COLOR)

    # Draw clefs/keys
    for clef in clefs_keys:
        clef.draw_onto_image(img, CLEF_COLOR)

    # Draw notes (noteheads with stems)
    for note in noteheads_with_stems:
        note.notehead.draw_onto_image(img, NOTE_COLOR)
        if note.stem is not None:
            note.stem.draw_onto_image(img, NOTE_COLOR)

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    y_offset = 30
    x_offset = 10

    cv2.putText(img, "Legend:", (x_offset, y_offset), font, font_scale, (255, 255, 255), thickness)
    y_offset += 25
    cv2.putText(img, "Green: Staffs", (x_offset, y_offset), font, font_scale, STAFF_COLOR, thickness)
    y_offset += 25
    cv2.putText(img, "Blue: Notes", (x_offset, y_offset), font, font_scale, NOTE_COLOR, thickness)
    y_offset += 25
    cv2.putText(img, "Red: Bar lines", (x_offset, y_offset), font, font_scale, BAR_LINE_COLOR, thickness)
    y_offset += 25
    cv2.putText(img, "Cyan: Clefs/Keys", (x_offset, y_offset), font, font_scale, CLEF_COLOR, thickness)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Comprehensive visualization saved to: {output_path}")
    print(f"Detected: {len(staffs)} staffs, {len(noteheads_with_stems)} notes, "
          f"{len(bar_line_boxes)} bar lines, {len(clefs_keys)} clefs/keys")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_comprehensive_visualization.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.replace(".png", "_annotated.png")

    create_comprehensive_visualization(image_path, output_path)
