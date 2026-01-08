"""
Extract individual measures using homr's detection + Orchestra-AI-2 style extraction.

This script replicates Orchestra-AI-2's approach:
1. Use homr to detect staff lines and bar lines
2. Create a colored "attached notes" visualization (flood fill from staffs)
3. Extract measures by finding ALL colored pixels in each measure's X range
4. Calculate individual top/bottom bounds for each measure
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# Import homr modules
from homr.resize import resize_image
from homr.debug import Debug
from homr.main import load_and_preprocess_predictions, predict_symbols
from homr.staff_detection import detect_staff
from homr.bar_line_detection import detect_bar_lines
from homr.note_detection import combine_noteheads_with_stems


def create_attached_notes_visualization(original_img, staffs):
    """
    Create a colored visualization where each staff's notes are colored uniquely.
    Uses flood fill (BFS) from staff lines to color all connected symbols.

    Args:
        original_img: Original grayscale/binary image
        staffs: List of Staff objects from homr

    Returns:
        Colored BGR image where each staff has a unique color
    """
    height, width = original_img.shape[:2]
    no_of_staffs = len(staffs)

    # Convert to BGR if needed
    if len(original_img.shape) == 2:
        vis_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
        gray = original_img
    else:
        vis_img = original_img.copy()
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Create mask of non-white pixels
    non_white_mask = gray < 255

    # Track which pixels have been assigned to a staff
    assigned = np.zeros((height, width), dtype=np.int32)  # 0 = unassigned

    # Generate distinct colors for each staff
    colors = []
    for i in range(no_of_staffs):
        hue = int(180 * i / no_of_staffs)
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))

    # Get staff line positions and centers
    staff_line_positions = []  # List of 5 Y positions per staff
    staff_centers = []
    for staff in staffs:
        # Get the actual 5 staff line Y positions from the grid
        # Average across all X positions for robustness
        line_ys = [[] for _ in range(5)]
        for grid_point in staff.grid:
            for line_idx, y_val in enumerate(grid_point.y):
                line_ys[line_idx].append(y_val)

        avg_line_ys = [np.mean(ys) for ys in line_ys]
        staff_line_positions.append(avg_line_ys)
        staff_centers.append(np.mean(avg_line_ys))

    # STEP 1: Seed ONLY the actual staff line pixels (not the entire staff range!)
    # This is critical to prevent staffs from bleeding into each other
    line_thickness = 3  # Seed pixels within ±3 pixels of each staff line
    for staff_idx, line_ys in enumerate(staff_line_positions):
        for line_y in line_ys:
            y_min = max(0, int(line_y - line_thickness))
            y_max = min(height - 1, int(line_y + line_thickness))
            for y in range(y_min, y_max + 1):
                for x in range(width):
                    if non_white_mask[y, x] and assigned[y, x] == 0:
                        assigned[y, x] = staff_idx + 1

    # STEP 2: Simultaneous BFS flood fill from all assigned pixels
    queue = deque()
    for y in range(height):
        for x in range(width):
            if assigned[y, x] > 0:
                queue.append((y, x, assigned[y, x] - 1))

    while queue:
        y, x, staff_idx = queue.popleft()

        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx

                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not non_white_mask[ny, nx]:
                    continue
                if assigned[ny, nx] != 0:
                    continue

                assigned[ny, nx] = staff_idx + 1
                queue.append((ny, nx, staff_idx))

    # STEP 3: Assign unattached symbols to nearest staff by Y position
    unassigned_mask = non_white_mask & (assigned == 0)
    component_visited = np.zeros((height, width), dtype=bool)

    for start_y in range(height):
        for start_x in range(width):
            if unassigned_mask[start_y, start_x] and not component_visited[start_y, start_x]:
                # Find connected component
                component_pixels = []
                comp_queue = deque([(start_y, start_x)])

                while comp_queue:
                    y, x = comp_queue.popleft()

                    if y < 0 or y >= height or x < 0 or x >= width:
                        continue
                    if component_visited[y, x]:
                        continue
                    if not unassigned_mask[y, x]:
                        continue

                    component_visited[y, x] = True
                    component_pixels.append((y, x))

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            comp_queue.append((y + dy, x + dx))

                # Assign to nearest staff based on average Y
                if component_pixels:
                    avg_y = sum(p[0] for p in component_pixels) / len(component_pixels)
                    min_dist = float('inf')
                    nearest_staff = 0

                    for staff_idx, center_y in enumerate(staff_centers):
                        dist = abs(avg_y - center_y)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_staff = staff_idx

                    for py, px in component_pixels:
                        assigned[py, px] = nearest_staff + 1

    # Color the visualization
    for y in range(height):
        for x in range(width):
            if assigned[y, x] > 0:
                staff_idx = assigned[y, x] - 1
                vis_img[y, x] = colors[staff_idx]

    return vis_img, colors


def extract_measures_for_staff(original_img, attached_img, staff, staff_idx, staff_color,
                               bar_lines, page_num, output_folder):
    """
    Extract measures using Orchestra-AI-2 style: search entire image height for colored pixels.

    Args:
        original_img: Original image
        attached_img: Colored attached notes visualization
        staff: Staff object
        staff_idx: Staff index (0-based)
        staff_color: BGR color tuple for this staff
        bar_lines: List of bar line bounding boxes
        page_num: Page number
        output_folder: Output directory

    Returns:
        List of (left, top, right, bottom) tuples for each measure
    """
    measures_folder = f"{output_folder}/measures"
    os.makedirs(measures_folder, exist_ok=True)

    height, width = attached_img.shape[:2]

    # Get staff boundaries
    staff_top = int(staff.min_y)
    staff_bottom = int(staff.max_y)
    staff_left = int(staff.min_x)
    staff_right = int(staff.max_x)

    # Filter bar lines that belong to this staff
    staff_bar_lines = []
    for bar_line in bar_lines:
        bar_center_y = bar_line.box[0][1]
        if staff_top <= bar_center_y <= staff_bottom:
            bar_x = int(bar_line.box[0][0])
            staff_bar_lines.append(bar_x)

    staff_bar_lines.sort()

    print(f"  Staff {staff_idx + 1}: Found {len(staff_bar_lines)} bar lines")

    # Add page edge boundaries as virtual bar lines
    # First measure should extend all the way to left edge (x=0)
    # Last measure should extend all the way to right edge (x=width)
    if not staff_bar_lines or staff_bar_lines[0] > 10:
        staff_bar_lines.insert(0, 0)
    if not staff_bar_lines or staff_bar_lines[-1] < width - 10:
        staff_bar_lines.append(width)

    measure_boxes = []
    b, g, r = staff_color
    color_tolerance = 30

    # Extract each measure
    for measure_idx in range(len(staff_bar_lines) - 1):
        left_bar = staff_bar_lines[measure_idx]
        right_bar = staff_bar_lines[measure_idx + 1]

        # Search ENTIRE image height for colored pixels in this X range
        left = max(0, left_bar)
        right = min(width - 1, right_bar)

        # Get region
        region = attached_img[:, left:right]

        if region.size == 0:
            continue

        # Find pixels matching this staff's color
        color_diff = np.abs(region.astype(np.int32) - np.array([b, g, r], dtype=np.int32))
        color_mask = np.all(color_diff < color_tolerance, axis=2)

        # Find rows with colored pixels
        colored_rows, colored_cols = np.where(color_mask)

        if len(colored_rows) == 0:
            continue

        # Find actual bounding box of colored pixels
        min_row = np.min(colored_rows)
        max_row = np.max(colored_rows)

        # Add padding
        padding = 10
        top = max(0, min_row - padding)
        bottom = min(height - 1, max_row + padding)

        # Extract measure
        measure_img = original_img[top:bottom, left:right]

        if measure_img.size == 0:
            continue

        # Convert to BGR if needed
        if len(measure_img.shape) == 2:
            measure_img = cv2.cvtColor(measure_img, cv2.COLOR_GRAY2BGR)

        # Store box coordinates
        measure_boxes.append((left, top, right, bottom))

        # Save
        filename = f"measure_{measure_idx + 1:04d}_p{page_num}_s{staff_idx + 1}_L{left}_T{top}_R{right}_B{bottom}.png"
        output_path = f"{measures_folder}/{filename}"
        cv2.imwrite(output_path, measure_img)

    print(f"    Extracted {len(measure_boxes)} measures")
    return measure_boxes


def create_measures_visualization(original_img, all_measure_boxes, output_path):
    """Create visualization showing all measure bounding boxes."""
    vis_img = original_img.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    for idx, (left, top, right, bottom) in enumerate(all_measure_boxes):
        color = colors[idx % len(colors)]
        cv2.rectangle(vis_img, (left, top), (right, bottom), color, 2)
        cv2.putText(vis_img, f"M{idx+1}", (left + 5, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, vis_img)
    print(f"✓ Saved measures visualization: {output_path}")


def process_image_with_measure_extraction(image_path, output_folder="output"):
    """Process image using homr detection + Orchestra-AI-2 style extraction."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}\n")

    # Extract page number
    page_num = 1
    basename = os.path.basename(image_path)
    if 'page_' in basename:
        try:
            page_num = int(basename.split('page_')[1].split('.')[0].split('_')[0])
        except:
            pass

    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load and preprocess with homr
    print("Step 1: Running homr detection...")
    predictions, debug = load_and_preprocess_predictions(
        image_path, enable_debug=False, enable_cache=False, use_gpu_inference=False
    )

    # Step 2: Predict symbols
    print("Step 2: Detecting symbols...")
    symbols = predict_symbols(debug, predictions)
    print(f"  Found {len(symbols.noteheads)} noteheads")
    print(f"  Found {len(symbols.bar_lines)} bar line candidates")

    # Step 3: Combine noteheads with stems
    print("Step 3: Combining noteheads with stems...")
    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    print(f"  Found {len(noteheads_with_stems)} complete notes")

    if len(noteheads_with_stems) == 0:
        print("❌ No noteheads found")
        return

    # Step 4: Detect bar lines
    print("Step 4: Detecting bar lines...")
    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    print(f"  Found {len(bar_line_boxes)} bar lines")

    # Step 5: Detect staffs
    print("Step 5: Detecting staffs...")
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    print(f"  Found {len(staffs)} staffs")

    if len(staffs) == 0:
        print("❌ No staffs found")
        return

    # Step 6: Create colored attached notes visualization
    print("Step 6: Creating colored attached notes visualization...")
    attached_img, staff_colors = create_attached_notes_visualization(predictions.original, staffs)
    attached_path = f"{output_folder}/attached_notes.png"
    cv2.imwrite(attached_path, attached_img)
    print(f"  ✓ Saved: {attached_path}")

    # Step 7: Extract measures for each staff
    print("Step 7: Extracting measures...")
    all_measure_boxes = []

    for staff_idx, staff in enumerate(staffs):
        measure_boxes = extract_measures_for_staff(
            predictions.original,
            attached_img,
            staff,
            staff_idx,
            staff_colors[staff_idx],
            bar_line_boxes,
            page_num,
            output_folder
        )
        all_measure_boxes.extend(measure_boxes)

    # Step 8: Create visualization
    print("Step 8: Creating visualization...")
    vis_path = f"{output_folder}/measures_visualization.png"
    create_measures_visualization(predictions.original, all_measure_boxes, vis_path)

    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Extracted {len(all_measure_boxes)} total measures")
    print(f"  Output saved to: {output_folder}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract individual measures using homr detection + Orchestra-AI-2 style'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--output', '-o', default='output_measures',
                        help='Output folder (default: output_measures)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    process_image_with_measure_extraction(args.input, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
