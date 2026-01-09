"""
Module for managing visualization outputs.
Handles clearing and saving detection visualizations to an output folder.
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from homr.bounding_boxes import DebugDrawable
from homr.type_definitions import NDArray


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

    def save_symbols_detection(self, clefs_keys: list[DebugDrawable]) -> None:
        """
        Save accidentals (sharps/flats/naturals) detection visualization.

        Args:
            clefs_keys: List of detected accidental bounding boxes
        """
        if self.original_image is None:
            return

        img = self.original_image.copy()

        # Create semi-transparent overlay for better visibility
        overlay = img.copy()

        for symbol in clefs_keys:
            # Draw with thick lines (5 pixels)
            import cv2 as cv2_local
            import numpy as np

            # Get bounding box points
            if hasattr(symbol, 'get_points'):
                pts = symbol.get_points()
                if pts is not None:
                    pts = pts.astype(np.int32)
                    # Draw thick outline
                    cv2_local.polylines(overlay, [pts], True, self.colors["symbols"], 5)
                    # Draw semi-transparent fill
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2_local.fillPoly(mask, [pts], 255)
                    overlay[mask > 0] = cv2_local.addWeighted(
                        overlay[mask > 0], 0.7,
                        np.full_like(overlay[mask > 0], self.colors["symbols"]), 0.3,
                        0
                    )
            else:
                # Fallback to regular drawing
                symbol.draw_onto_image(overlay, self.colors["symbols"])

        # Blend overlay with original
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        output_path = os.path.join(self.output_dir, f"{self.base_name}_accidentals.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved accidentals detection: {output_path} ({len(clefs_keys)} accidentals)")

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
