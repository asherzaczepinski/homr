"""
Module for managing visualization outputs.
Handles clearing and saving detection visualizations to an output folder.
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from homr.core.bounding_boxes import DebugDrawable
from homr.core.type_definitions import NDArray


class VisualizationOutput:
    """Manages output folder and saves specific visualization types."""

    def __init__(self, image_path: str, output_dir: str = "output"):
        """
        Initialize the visualization output manager.

        Args:
            image_path: Path to the input image
            output_dir: Name of the output directory (default: "output")
        """
        self.image_path = image_path

        # Create output directory path in the current working directory
        self.output_dir = output_dir

        # Get base filename without extension
        self.base_name = Path(image_path).stem

        # Clear and create output directory
        self._setup_output_dir()

        # Store original image for drawing
        self.original_image: NDArray | None = None

        # Colors for different detection types
        self.colors = {
            "staff": (0, 255, 0),      # Green
            "measures": (0, 0, 255),   # Red
            "notes": (255, 0, 0),      # Blue
            "symbols": (255, 255, 0),  # Cyan
        }

    def _setup_output_dir(self) -> None:
        """Clear and create the output directory."""
        # Remove existing output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        # Create fresh output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def set_original_image(self, image: NDArray) -> None:
        """Store the original image for visualization."""
        self.original_image = image.copy()

    def save_staff_detection(self, staffs: list[DebugDrawable]) -> None:
        """
        Save staff detection visualization.

        Args:
            staffs: List of detected staff bounding boxes
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()
        for staff in staffs:
            staff.draw_onto_image(img, self.colors["staff"])

        output_path = os.path.join(self.output_dir, f"{self.base_name}_staffs.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved staff detection: {output_path}")

    def save_measures_detection(self, bar_lines: list[DebugDrawable]) -> None:
        """
        Save measures (bar lines) detection visualization.

        Args:
            bar_lines: List of detected bar line bounding boxes
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()
        for bar_line in bar_lines:
            bar_line.draw_onto_image(img, self.colors["measures"])

        output_path = os.path.join(self.output_dir, f"{self.base_name}_measures.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved measures detection: {output_path}")

    def save_notes_detection(self, noteheads_with_stems: list) -> None:
        """
        Save notes detection visualization.

        Args:
            noteheads_with_stems: List of detected notes with stems
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()
        for note in noteheads_with_stems:
            # Only draw noteheads, not stems
            note.notehead.draw_onto_image(img, self.colors["notes"])

        output_path = os.path.join(self.output_dir, f"{self.base_name}_notes.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved notes detection: {output_path}")

    def classify_accidental(self, symbol) -> str:
        """
        Classify an accidental as sharp, flat, or natural based on geometry.

        Args:
            symbol: Bounding box of the accidental

        Returns:
            "sharp", "flat", or "natural"
        """
        width = symbol.size[0]
        height = symbol.size[1]

        # Avoid division by zero
        if width == 0:
            return "natural"

        aspect_ratio = height / width

        # Classification based on aspect ratio:
        # Sharp (#): Very tall and thin, aspect ratio > 2.5
        # Flat (♭): Tall but wider, aspect ratio 1.8 - 2.5
        # Natural (♮): More square, aspect ratio < 1.8
        if aspect_ratio > 2.5:
            return "sharp"
        elif aspect_ratio > 1.8:
            return "flat"
        else:
            return "natural"

    def save_symbols_detection(self, clefs_keys: list[DebugDrawable], accidental_detections: list = None) -> None:
        """
        Save accidentals (sharps/flats/naturals) detection visualization.
        Uses Orchestra-AI-2 style when available, or falls back to geometry.

        Args:
            clefs_keys: List of detected accidental bounding boxes
            accidental_detections: Optional list of AccidentalDetection objects from Orchestra-AI-2
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # If we have Orchestra-AI-2 detections, use their native drawing style
        if accidental_detections and len(accidental_detections) > 0:
            print("✓ Using Orchestra-AI-2 detections for visualization")

            # Draw using Orchestra-AI-2 style (labels with confidence)
            try:
                # Draw each detection with labels and confidence scores
                for det in accidental_detections:
                    # Get bounding box coordinates from RotatedBoundingBox
                    top_left = det.bbox.top_left
                    bottom_right = det.bbox.bottom_right
                    x1, y1 = int(top_left[0]), int(top_left[1])
                    x2, y2 = int(bottom_right[0]), int(bottom_right[1])

                    color = (0, 255, 255)  # Cyan in BGR
                    thickness = 2

                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                    # Draw green reference line based on accidental type
                    green_color = (0, 255, 0)  # Green in BGR
                    green_thickness = 1

                    class_name = det.class_name.lower()
                    height = y2 - y1

                    # Sharp and Natural: line at middle (50%)
                    if 'sharp' in class_name or 'natural' in class_name:
                        line_y = int(y1 + height * 0.5)
                        cv2.line(img, (x1, line_y), (x2, line_y), green_color, green_thickness)
                    # Flat: line at bottom 1/4 (75% from top)
                    elif 'flat' in class_name:
                        line_y = int(y1 + height * 0.75)
                        cv2.line(img, (x1, line_y), (x2, line_y), green_color, green_thickness)

                    # Draw label with class name and confidence
                    # Shorten class name for display
                    short_name = det.class_name.replace('accidental', '').replace('key', 'key')
                    label = f"{short_name} {det.confidence:.2f}"

                    # Draw label background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                    # Draw filled rectangle for label background
                    cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)

                    # Draw text
                    cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), font_thickness)

                # Count by type
                counts = {}
                for det in accidental_detections:
                    counts[det.class_name] = counts.get(det.class_name, 0) + 1

                output_path = os.path.join(self.output_dir, f"{self.base_name}_accidentals.png")
                cv2.imwrite(output_path, img)

                # Format count summary
                count_str = ", ".join([f"{count} {name}" for name, count in counts.items()])
                print(f"✓ Saved accidentals detection: {output_path} ({len(accidental_detections)} total: {count_str})")

            except Exception as e:
                print(f"Warning: Could not use Orchestra-AI-2 drawing style: {e}")
                print("Falling back to basic visualization")
                self._draw_fallback_visualization(img, clefs_keys)
        else:
            # Fallback to geometric classification
            print("⚠ FALLBACK: Using geometric classification for accidental visualization (Orchestra-AI-2 not available)")
            self._draw_fallback_visualization(img, clefs_keys)

    def _draw_fallback_visualization(self, img: NDArray, clefs_keys: list[DebugDrawable]) -> None:
        """Draw fallback visualization using geometric classification."""
        # Create semi-transparent overlay for better visibility
        overlay = img.copy()

        # Color mapping for accidentals
        accidental_colors = {
            "sharp": (0, 0, 255),      # Red (BGR)
            "flat": (255, 200, 0),     # Light Blue (BGR)
            "natural": (0, 255, 0),    # Green (BGR)
        }

        # Count each type
        counts = {"sharp": 0, "flat": 0, "natural": 0}

        for symbol in clefs_keys:
            # Classify the accidental
            acc_type = self.classify_accidental(symbol)
            counts[acc_type] += 1
            color = accidental_colors[acc_type]

            # Draw with thick lines (5 pixels)
            import cv2 as cv2_local
            import numpy as np

            # Get bounding box points
            if hasattr(symbol, 'get_points'):
                pts = symbol.get_points()
                if pts is not None:
                    pts = pts.astype(np.int32)
                    # Draw thick outline
                    cv2_local.polylines(overlay, [pts], True, color, 5)
                    # Draw semi-transparent fill
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2_local.fillPoly(mask, [pts], 255)
                    overlay[mask > 0] = cv2_local.addWeighted(
                        overlay[mask > 0], 0.7,
                        np.full_like(overlay[mask > 0], color), 0.3,
                        0
                    )
            else:
                # Fallback to regular drawing
                symbol.draw_onto_image(overlay, color)

        # Blend overlay with original
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_accidentals.png")
        cv2.imwrite(output_path, img)

        # Format count summary
        count_str = ", ".join([f"{count} {name}" for name, count in counts.items()])
        print(f"✓ Saved accidentals detection: {output_path} ({len(clefs_keys)} total: {count_str})")

    def save_accidental_positions_analysis(self, staffs: list, accidental_detections: list) -> None:
        """
        Show which red line each accidental belongs to, with measure-by-measure line calculation.
        Draws lines that follow staff curvature by recalculating positions for each measure.

        Args:
            staffs: List of detected Staff objects
            accidental_detections: List of AccidentalDetection objects from Orchestra-AI-2
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # Color for the position lines (red for non-staff, pink for staff)
        line_color = (0, 0, 255)  # Red in BGR
        pink_color = (203, 192, 255)  # Pink in BGR (for actual staff lines)
        line_thickness = 1

        # Color for accidental markers (green)
        green_color = (0, 255, 0)  # Green in BGR
        green_section_length = 10  # Length on each side of the marker

        # Get full image width
        img_width = img.shape[1]

        # Store all line positions for accidental matching (x, y) tuples
        all_line_positions = []  # List of (x_center, y_position, line_type) for each segment

        for staff_idx, staff in enumerate(staffs):
            unit_size = staff.average_unit_size
            grid_points = staff.grid

            # Process measure by measure (grid point to grid point)
            for grid_idx in range(len(grid_points) - 1):
                curr_point = grid_points[grid_idx]
                next_point = grid_points[grid_idx + 1]

                # X range for this measure
                x_start = int(curr_point.x)
                x_end = int(next_point.x)
                x_center = (x_start + x_end) / 2

                # Get the 5 staff line Y positions at start and end of this measure
                curr_staff_lines = [curr_point.y[i] for i in range(5)]
                next_staff_lines = [next_point.y[i] for i in range(5)]

                # DRAW THE 5 ACTUAL STAFF LINES IN PINK (curved to follow staff)
                for line_idx in range(5):
                    y_start = int(curr_staff_lines[line_idx])
                    y_end = int(next_staff_lines[line_idx])

                    # Draw line segment for this measure
                    cv2.line(img, (x_start, y_start), (x_end, y_end), pink_color, line_thickness)

                    # Store position for accidental matching (use center Y of this segment)
                    y_center = (y_start + y_end) / 2
                    all_line_positions.append((x_center, y_center, 'staff'))

                # DRAW THE 4 SPACES BETWEEN STAFF LINES
                for space_idx in range(4):
                    # Calculate space position between lines at start and end
                    y_start = int((curr_staff_lines[space_idx] + curr_staff_lines[space_idx + 1]) / 2)
                    y_end = int((next_staff_lines[space_idx] + next_staff_lines[space_idx + 1]) / 2)

                    # Draw line segment
                    cv2.line(img, (x_start, y_start), (x_end, y_end), line_color, line_thickness)

                    # Store position
                    y_center = (y_start + y_end) / 2
                    all_line_positions.append((x_center, y_center, 'space'))

                # DRAW LEDGER LINES ABOVE THE STAFF
                top_line_start = curr_staff_lines[0]
                top_line_end = next_staff_lines[0]

                for i in range(1, 13):  # 12 half-units above
                    y_start = int(top_line_start - (i * unit_size / 2))
                    y_end = int(top_line_end - (i * unit_size / 2))

                    if 0 <= y_start < img.shape[0] and 0 <= y_end < img.shape[0]:
                        cv2.line(img, (x_start, y_start), (x_end, y_end), line_color, line_thickness)
                        y_center = (y_start + y_end) / 2
                        all_line_positions.append((x_center, y_center, 'ledger'))

                # DRAW LEDGER LINES BELOW THE STAFF (only for last staff)
                if staff_idx == len(staffs) - 1:
                    bottom_line_start = curr_staff_lines[4]
                    bottom_line_end = next_staff_lines[4]

                    for i in range(1, 13):  # 12 half-units below
                        y_start = int(bottom_line_start + (i * unit_size / 2))
                        y_end = int(bottom_line_end + (i * unit_size / 2))

                        if 0 <= y_start < img.shape[0] and 0 <= y_end < img.shape[0]:
                            cv2.line(img, (x_start, y_start), (x_end, y_end), line_color, line_thickness)
                            y_center = (y_start + y_end) / 2
                            all_line_positions.append((x_center, y_center, 'ledger'))

        # Now process each accidental detection
        if accidental_detections:
            for detection in accidental_detections:
                # Get the reference position from the accidental
                top_left = detection.bbox.top_left
                bottom_right = detection.bbox.bottom_right
                x1, y1 = top_left[0], top_left[1]
                x2, y2 = bottom_right[0], bottom_right[1]
                height = y2 - y1

                class_name = detection.class_name.lower()

                # Calculate the reference Y position based on accidental type
                if 'sharp' in class_name or 'natural' in class_name:
                    ref_y = y1 + height * 0.5  # Middle
                elif 'flat' in class_name:
                    ref_y = y1 + height * 0.75  # Bottom 1/4
                else:
                    ref_y = (y1 + y2) / 2  # Default to middle

                # Get the horizontal center of the accidental
                center_x = (x1 + x2) / 2

                # Find the closest line position considering both X and Y proximity
                # Filter to lines near this X position first
                nearby_lines = [(x, y, t) for x, y, t in all_line_positions if abs(x - center_x) < 100]

                if nearby_lines:
                    # Find the line with the closest Y position
                    closest_line = min(nearby_lines, key=lambda pos: abs(pos[1] - ref_y))
                    closest_x, closest_y, line_type = closest_line

                    # Draw green section on the line where this accidental belongs
                    left_x = max(0, int(center_x - green_section_length))
                    right_x = min(img_width - 1, int(center_x + green_section_length))

                    cv2.line(img, (left_x, int(closest_y)), (right_x, int(closest_y)), green_color, 2)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_accidentals2.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved accidental positions analysis: {output_path}")

    def save_accidental_affected_notes(self, staffs: list, accidental_detections: list, noteheads_with_stems: list) -> None:
        """
        Circle notes that are affected by accidentals (sharps/flats).
        Accidentals affect notes on the same line, 7 half-lines up, 14 half-lines up, etc.
        Naturals cancel accidentals. Effects carry across the measure but not across staffs.

        Args:
            staffs: List of detected Staff objects
            accidental_detections: List of AccidentalDetection objects from Orchestra-AI-2
            noteheads_with_stems: List of detected noteheads
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # Process each staff independently
        for staff_idx, staff in enumerate(staffs):
            unit_size = staff.average_unit_size
            grid_points = staff.grid

            # Build line positions for this staff (measure by measure)
            staff_line_positions = {}  # Maps x_position -> list of y positions for lines

            for grid_idx in range(len(grid_points) - 1):
                curr_point = grid_points[grid_idx]
                next_point = grid_points[grid_idx + 1]

                x_start = curr_point.x
                x_end = next_point.x
                x_center = (x_start + x_end) / 2

                # Get staff lines at this position
                curr_staff_lines = [curr_point.y[i] for i in range(5)]
                next_staff_lines = [next_point.y[i] for i in range(5)]

                # Calculate all line positions (staff lines, spaces, ledgers)
                all_lines_here = []

                # 5 staff lines
                for i in range(5):
                    y_pos = (curr_staff_lines[i] + next_staff_lines[i]) / 2
                    all_lines_here.append(y_pos)

                # 4 spaces
                for i in range(4):
                    y_pos = ((curr_staff_lines[i] + curr_staff_lines[i + 1]) / 2 +
                             (next_staff_lines[i] + next_staff_lines[i + 1]) / 2) / 2
                    all_lines_here.append(y_pos)

                # Ledger lines above (12 half-steps)
                top_line = (curr_staff_lines[0] + next_staff_lines[0]) / 2
                for i in range(1, 13):
                    y_pos = top_line - (i * unit_size / 2)
                    all_lines_here.append(y_pos)

                # Ledger lines below (12 half-steps)
                bottom_line = (curr_staff_lines[4] + next_staff_lines[4]) / 2
                for i in range(1, 13):
                    y_pos = bottom_line + (i * unit_size / 2)
                    all_lines_here.append(y_pos)

                # Sort by Y position and store
                all_lines_here.sort()
                staff_line_positions[x_center] = all_lines_here

            # Track active accidentals: maps line_index -> accidental_type ('sharp', 'flat', 'natural')
            active_accidentals = {}

            # Get accidentals for this staff, sorted by X position (left to right)
            staff_accidentals = []
            if accidental_detections:
                for det in accidental_detections:
                    acc_center_x = (det.bbox.top_left[0] + det.bbox.bottom_right[0]) / 2
                    acc_y = det.bbox.top_left[1] + (det.bbox.bottom_right[1] - det.bbox.top_left[1]) * 0.5
                    if 'flat' in det.class_name.lower():
                        acc_y = det.bbox.top_left[1] + (det.bbox.bottom_right[1] - det.bbox.top_left[1]) * 0.75

                    # Check if this accidental is on this staff
                    staff_y_min = min(grid_points[0].y)
                    staff_y_max = max(grid_points[0].y)
                    if staff_y_min - 50 <= acc_y <= staff_y_max + 50:
                        staff_accidentals.append((acc_center_x, acc_y, det.class_name))

            staff_accidentals.sort(key=lambda x: x[0])  # Sort by X position

            # Process accidentals from left to right
            for acc_x, acc_y, acc_class in staff_accidentals:
                # Find which line this accidental is on
                closest_x_pos = min(staff_line_positions.keys(), key=lambda x: abs(x - acc_x))
                lines_at_pos = staff_line_positions[closest_x_pos]
                line_idx = min(range(len(lines_at_pos)), key=lambda i: abs(lines_at_pos[i] - acc_y))

                # Determine accidental type
                acc_type = None
                if 'sharp' in acc_class.lower():
                    acc_type = 'sharp'
                elif 'flat' in acc_class.lower():
                    acc_type = 'flat'
                elif 'natural' in acc_class.lower():
                    acc_type = 'natural'

                if acc_type:
                    # Mark this line and all octave equivalents (7 and 14 half-steps)
                    for offset in [0, 7, 14, -7, -14]:
                        affected_line = line_idx + offset
                        if 0 <= affected_line < len(lines_at_pos):
                            if acc_type == 'natural':
                                # Natural cancels out the accidental - REMOVE it completely
                                if affected_line in active_accidentals:
                                    del active_accidentals[affected_line]
                            else:
                                # Sharp or flat - mark it
                                active_accidentals[affected_line] = acc_type

            # Track which accidentals affect which X ranges
            accidental_ranges = {}  # Maps (line_idx, acc_type) -> start_x position

            # Re-process accidentals to track their X positions
            current_accidentals = {}  # Maps line_idx -> (acc_type, start_x)
            for acc_x, acc_y, acc_class in staff_accidentals:
                closest_x_pos = min(staff_line_positions.keys(), key=lambda x: abs(x - acc_x))
                lines_at_pos = staff_line_positions[closest_x_pos]
                line_idx = min(range(len(lines_at_pos)), key=lambda i: abs(lines_at_pos[i] - acc_y))

                acc_type = None
                if 'sharp' in acc_class.lower():
                    acc_type = 'sharp'
                elif 'flat' in acc_class.lower():
                    acc_type = 'flat'
                elif 'natural' in acc_class.lower():
                    acc_type = 'natural'

                if acc_type:
                    for offset in [0, 7, 14, -7, -14]:
                        affected_line = line_idx + offset
                        if 0 <= affected_line < len(lines_at_pos):
                            if acc_type == 'natural':
                                # Natural cancels - remove from tracking
                                if affected_line in current_accidentals:
                                    del current_accidentals[affected_line]
                            else:
                                # Sharp or flat - start affecting from this X position
                                current_accidentals[affected_line] = (acc_type, acc_x)

            # Now process notes and circle affected ones
            for note in noteheads_with_stems:
                notehead = note.notehead
                note_center_x = notehead.center[0]
                note_center_y = notehead.center[1]

                # Check if note is on this staff
                staff_y_min = min(grid_points[0].y)
                staff_y_max = max(grid_points[0].y)
                if not (staff_y_min - 50 <= note_center_y <= staff_y_max + 50):
                    continue

                # Find which line this note is on
                closest_x_pos = min(staff_line_positions.keys(), key=lambda x: abs(x - note_center_x))
                lines_at_pos = staff_line_positions[closest_x_pos]
                note_line_idx = min(range(len(lines_at_pos)), key=lambda i: abs(lines_at_pos[i] - note_center_y))

                # Check if this line is affected by an accidental AND note is after the accidental
                if note_line_idx in current_accidentals:
                    acc_type, acc_x = current_accidentals[note_line_idx]

                    # Only circle if note is AFTER the accidental
                    if note_center_x > acc_x:
                        # Choose color based on accidental type
                        if acc_type == 'sharp':
                            color = (0, 0, 255)  # Red in BGR
                        elif acc_type == 'flat':
                            color = (255, 200, 0)  # Light blue in BGR
                        else:
                            color = (0, 255, 0)  # Green (shouldn't happen)

                        # Draw ellipse around the note (3 extra pixels in radius)
                        center = (int(notehead.center[0]), int(notehead.center[1]))
                        axes = (int(notehead.size[0] / 2 + 6), int(notehead.size[1] / 2 + 6))
                        angle = notehead.angle if hasattr(notehead, 'angle') else 0

                        cv2.ellipse(img, center, axes, angle, 0, 360, color, 2)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_accidental_effects.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved accidental effects visualization: {output_path}")

    def save_musicxml(self, xml_content: str) -> str:
        """
        Save the MusicXML file to the output directory.

        Args:
            xml_content: The XML content to save

        Returns:
            Path to the saved XML file
        """
        output_path = os.path.join(self.output_dir, f"{self.base_name}.musicxml")
        with open(output_path, 'w') as f:
            f.write(xml_content)
        print(f"✓ Saved MusicXML: {output_path}")
        return output_path

    def save_note_positions_grid(self, staffs: list) -> None:
        """
        Save visualization showing all possible note positions on staffs.
        Draws red lines at every line and space position where notes could be.

        Args:
            staffs: List of detected Staff objects with grid information
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # Color for the position lines (red)
        line_color = (0, 0, 255)  # Red in BGR
        line_thickness = 1

        # Get full image width
        img_width = img.shape[1]

        # Keep track of all line Y positions drawn to avoid duplicates
        drawn_lines = set()

        for staff_idx, staff in enumerate(staffs):
            # Get average unit size (distance between staff lines) for THIS staff
            unit_size = staff.average_unit_size

            # Get the 5 actual staff line Y positions (average across the staff width)
            # Calculate average Y for each of the 5 lines
            staff_line_positions = []
            for line_idx in range(5):
                y_values = [grid_point.y[line_idx] for grid_point in staff.grid]
                avg_y = sum(y_values) / len(y_values)
                staff_line_positions.append(avg_y)

            # DRAW THE 5 ACTUAL STAFF LINES at their exact positions
            # ALWAYS draw these no matter what (no duplicate check)
            for staff_line_y in staff_line_positions:
                y = int(staff_line_y)
                if 0 <= y < img.shape[0]:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                    drawn_lines.add(y)  # Track for other calculated lines

            # DRAW THE 4 SPACES BETWEEN STAFF LINES
            for i in range(4):  # 4 spaces between 5 lines
                # Space is halfway between line i and line i+1
                space_y = (staff_line_positions[i] + staff_line_positions[i + 1]) / 2
                y = int(space_y)
                if 0 <= y < img.shape[0] and y not in drawn_lines:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                    drawn_lines.add(y)

            # DRAW LINES ABOVE THE STAFF
            top_line = staff_line_positions[0]

            # Check if there's a staff above this one
            if staff_idx > 0:
                prev_staff = staffs[staff_idx - 1]
                prev_bottom_line = sum([gp.y[4] for gp in prev_staff.grid]) / len(prev_staff.grid)

                # If staffs are close, draw lines in the gap between them
                gap_start = prev_bottom_line
                gap_end = top_line
                gap_size = gap_end - gap_start

                # Only draw gap lines if there's a reasonable gap
                if gap_size > unit_size:
                    # Draw lines at half-unit intervals in the gap
                    num_lines = int(gap_size / (unit_size / 2))
                    for i in range(1, num_lines):
                        y = int(gap_start + (i * gap_size / num_lines))
                        if 0 <= y < img.shape[0] and y not in drawn_lines:
                            cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                            drawn_lines.add(y)
                else:
                    # Gap is small, just draw one line in the middle
                    y = int((gap_start + gap_end) / 2)
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        drawn_lines.add(y)
            else:
                # First staff - draw lines above normally
                for i in range(1, 13):  # 12 half-units above (6 full units)
                    y = int(top_line - (i * unit_size / 2))
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        drawn_lines.add(y)

            # DRAW LINES BELOW THE STAFF (only for the last staff)
            if staff_idx == len(staffs) - 1:
                bottom_line = staff_line_positions[4]
                for i in range(1, 13):  # 12 half-units below (6 full units)
                    y = int(bottom_line + (i * unit_size / 2))
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        drawn_lines.add(y)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_note_positions.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved note positions grid: {output_path}")

    def save_note_splitting_analysis(self, staffs: list, noteheads_with_stems: list) -> None:
        """
        Analyze grouped noteheads and show which red line each note belongs to.
        For each chord/group, splits it into individual notes and marks each with yellow.

        Args:
            staffs: List of detected Staff objects
            noteheads_with_stems: List of detected noteheads (may be grouped)
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # Color for the position lines (red)
        line_color = (0, 0, 255)  # Red in BGR
        line_thickness = 1

        # Color for note markers (yellow)
        marker_color = (0, 255, 255)  # Yellow in BGR
        marker_radius = 3

        # Get full image width
        img_width = img.shape[1]

        # Calculate and draw all red lines, storing their Y positions
        all_red_lines = []  # List of Y positions
        drawn_lines = set()

        for staff_idx, staff in enumerate(staffs):
            unit_size = staff.average_unit_size

            # Get the 5 actual staff line Y positions
            staff_line_positions = []
            for line_idx in range(5):
                y_values = [grid_point.y[line_idx] for grid_point in staff.grid]
                avg_y = sum(y_values) / len(y_values)
                staff_line_positions.append(avg_y)

            # DRAW THE 5 ACTUAL STAFF LINES
            for staff_line_y in staff_line_positions:
                y = int(staff_line_y)
                if 0 <= y < img.shape[0]:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                    all_red_lines.append(y)
                    drawn_lines.add(y)

            # DRAW THE 4 SPACES BETWEEN STAFF LINES
            for i in range(4):
                space_y = (staff_line_positions[i] + staff_line_positions[i + 1]) / 2
                y = int(space_y)
                if 0 <= y < img.shape[0] and y not in drawn_lines:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                    all_red_lines.append(y)
                    drawn_lines.add(y)

            # DRAW LINES ABOVE THE STAFF
            top_line = staff_line_positions[0]

            if staff_idx > 0:
                prev_staff = staffs[staff_idx - 1]
                prev_bottom_line = sum([gp.y[4] for gp in prev_staff.grid]) / len(prev_staff.grid)
                gap_start = prev_bottom_line
                gap_end = top_line
                gap_size = gap_end - gap_start

                if gap_size > unit_size:
                    num_lines = int(gap_size / (unit_size / 2))
                    for i in range(1, num_lines):
                        y = int(gap_start + (i * gap_size / num_lines))
                        if 0 <= y < img.shape[0] and y not in drawn_lines:
                            cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                            all_red_lines.append(y)
                            drawn_lines.add(y)
                else:
                    y = int((gap_start + gap_end) / 2)
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        all_red_lines.append(y)
                        drawn_lines.add(y)
            else:
                for i in range(1, 13):
                    y = int(top_line - (i * unit_size / 2))
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        all_red_lines.append(y)
                        drawn_lines.add(y)

            # DRAW LINES BELOW THE STAFF (only for last staff)
            if staff_idx == len(staffs) - 1:
                bottom_line = staff_line_positions[4]
                for i in range(1, 13):
                    y = int(bottom_line + (i * unit_size / 2))
                    if 0 <= y < img.shape[0] and y not in drawn_lines:
                        cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                        all_red_lines.append(y)
                        drawn_lines.add(y)

        # Sort red lines by Y position for easier searching
        all_red_lines.sort()

        # Color for green indicator sections
        green_color = (0, 255, 0)  # Green in BGR
        green_section_length = 10  # Length on each side of the marker

        # Store yellow marker positions to draw green sections later
        yellow_markers = []  # List of (x, y) tuples

        # Now analyze each notehead group and split it
        for notehead_group in noteheads_with_stems:
            notehead = notehead_group.notehead

            # Get bounding box of the notehead group
            top_y = int(notehead.top_left[1])
            bottom_y = int(notehead.bottom_right[1])
            center_x = int(notehead.center[0])

            # Find all red lines that fall within this notehead's vertical range
            lines_in_range = [line_y for line_y in all_red_lines if top_y <= line_y <= bottom_y]

            if len(lines_in_range) == 0:
                # Single note - place one yellow marker at center
                y_pos = int(notehead.center[1])
                cv2.circle(img, (center_x, y_pos), marker_radius, marker_color, -1)
                yellow_markers.append((center_x, y_pos))
            else:
                # Calculate how many FULL LINE DISTANCES are spanned
                # A note takes up 2 half-lines (one full line distance)
                # If we have N half-lines (red lines), we have N-1 intervals between them
                # Number of notes = number of lines we cross (approximately)

                # The height of the group in pixels
                height_px = bottom_y - top_y

                # Average distance between consecutive red lines in this range
                if len(lines_in_range) > 1:
                    avg_spacing = (lines_in_range[-1] - lines_in_range[0]) / (len(lines_in_range) - 1)
                else:
                    # Use general spacing from nearby lines
                    avg_spacing = 16  # fallback

                # A note takes up approximately 2 half-lines = 1 full line distance
                # So number of notes = height / (2 * half_line_spacing)
                # Since half_line_spacing = avg_spacing, a full line distance = 2 * avg_spacing
                note_height = 2 * avg_spacing
                if note_height > 0:
                    num_notes = max(1, round(height_px / note_height))
                else:
                    num_notes = 1

                # Place yellow markers evenly distributed through the chord
                for i in range(num_notes):
                    # Calculate Y position for this note
                    # Distribute evenly from top to bottom
                    if num_notes == 1:
                        y_pos = int((top_y + bottom_y) / 2)
                    else:
                        y_pos = int(top_y + (i + 0.5) * height_px / num_notes)

                    cv2.circle(img, (center_x, y_pos), marker_radius, marker_color, -1)
                    yellow_markers.append((center_x, y_pos))

        # Now draw green sections on red lines where yellow markers touch
        for marker_x, marker_y in yellow_markers:
            # Find the closest red line to this marker
            closest_line_y = min(all_red_lines, key=lambda line_y: abs(line_y - marker_y))

            # Draw green sections on the left and right of the marker
            left_x = max(0, marker_x - green_section_length)
            right_x = min(img_width - 1, marker_x + green_section_length)

            cv2.line(img, (left_x, closest_line_y), (right_x, closest_line_y), green_color, line_thickness)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_note_splitting.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved note splitting analysis: {output_path}")

    def save_everything(self, staffs, noteheads_with_stems):
        """Show all notes with yellow markers: individual notes assigned to closest line, grouped notes split."""
        img = self.original_image.copy()
        if img is None:
            print(f"✗ Could not load image for everything visualization")
            return

        # Draw red lines for note positions
        line_color = (0, 0, 255)  # Red in BGR
        line_thickness = 1
        img_width = img.shape[1]

        all_staff_lines = []
        all_half_lines = []

        for staff in staffs:
            # Get the 5 actual staff line Y positions (average across the staff width)
            staff_line_positions = []
            for line_idx in range(5):
                y_values = [grid_point.y[line_idx] for grid_point in staff.grid]
                avg_y = sum(y_values) / len(y_values)
                staff_line_positions.append(avg_y)

            all_staff_lines.append(staff_line_positions)

            # Draw lines on staff lines
            for staff_line_y in staff_line_positions:
                y = int(staff_line_y)
                if 0 <= y < img.shape[0]:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                all_half_lines.append(staff_line_y)

            # Draw lines between staff lines (spaces)
            for i in range(4):
                mid_y = (staff_line_positions[i] + staff_line_positions[i + 1]) / 2
                y = int(mid_y)
                if 0 <= y < img.shape[0]:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                all_half_lines.append(mid_y)

        # Handle gaps between staffs
        for i in range(len(all_staff_lines) - 1):
            bottom_of_current = all_staff_lines[i][-1]
            top_of_next = all_staff_lines[i + 1][0]
            gap = top_of_next - bottom_of_current

            if gap > 5:
                mid_y = (bottom_of_current + top_of_next) / 2
                y = int(mid_y)
                if 0 <= y < img.shape[0]:
                    cv2.line(img, (0, y), (img_width - 1, y), line_color, line_thickness)
                all_half_lines.append(mid_y)

        all_half_lines = sorted(all_half_lines)

        # Group nearby noteheads (same logic as note_splitting_analysis)
        marker_color = (0, 255, 255)  # Yellow in BGR
        marker_radius = 3

        noteheads = [n.notehead for n in noteheads_with_stems]
        grouped = []
        used = set()

        for i, note in enumerate(noteheads):
            if i in used:
                continue
            group = [note]
            used.add(i)

            for j, other in enumerate(noteheads):
                if j in used:
                    continue
                if abs(note.center[0] - other.center[0]) < 20:
                    group.append(other)
                    used.add(j)

            grouped.append(group)

        # Process each group
        for group in grouped:
            if len(group) == 1:
                # Individual note - assign to closest half-line
                note = group[0]
                center_y = note.center[1]

                # Find closest half-line
                closest_line = min(all_half_lines, key=lambda y: abs(y - center_y))
                center_x = int(note.center[0])
                y_pos = int(closest_line)

                cv2.circle(img, (center_x, y_pos), marker_radius, marker_color, -1)
            else:
                # Grouped notes (chord) - use splitting logic
                top_y = min(n.top_left[1] for n in group)
                bottom_y = max(n.bottom_right[1] for n in group)
                center_x = int(np.mean([n.center[0] for n in group]))

                # Find lines in range
                lines_in_range = [y for y in all_half_lines if top_y <= y <= bottom_y]

                if len(lines_in_range) > 1:
                    avg_spacing = (lines_in_range[-1] - lines_in_range[0]) / (len(lines_in_range) - 1)
                else:
                    avg_spacing = 16

                height_px = bottom_y - top_y
                note_height = 2 * avg_spacing  # A note = 2 half-lines
                num_notes = max(1, round(height_px / note_height))

                # Place yellow markers
                for i in range(num_notes):
                    if num_notes == 1:
                        y_pos = int((top_y + bottom_y) / 2)
                    else:
                        y_pos = int(top_y + (i + 0.5) * height_px / num_notes)
                    cv2.circle(img, (center_x, y_pos), marker_radius, marker_color, -1)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_everything.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved everything visualization: {output_path}")

    def get_summary(self) -> str:
        """Get a summary of the output directory."""
        return f"\nAll outputs saved to: {self.output_dir}/"
