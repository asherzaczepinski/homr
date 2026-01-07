"""
Module for managing visualization outputs.
Handles clearing and saving detection visualizations to an output folder.
"""
import os
import shutil
from pathlib import Path

import cv2

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

        # Create output directory path relative to the image location
        image_dir = os.path.dirname(os.path.abspath(image_path))
        self.output_dir = os.path.join(image_dir, output_dir)

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
        """Clear and recreate the output directory."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
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
            note.notehead.draw_onto_image(img, self.colors["notes"])
            if note.stem is not None:
                note.stem.draw_onto_image(img, self.colors["notes"])

        output_path = os.path.join(self.output_dir, f"{self.base_name}_notes.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved notes detection: {output_path}")

    def save_symbols_detection(self, clefs_keys: list[DebugDrawable]) -> None:
        """
        Save symbols (clefs/keys) detection visualization.

        Args:
            clefs_keys: List of detected symbol bounding boxes
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

        output_path = os.path.join(self.output_dir, f"{self.base_name}_symbols.png")
        cv2.imwrite(output_path, img)
        print(f"✓ Saved symbols detection: {output_path} ({len(clefs_keys)} symbols)")

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

    def get_summary(self) -> str:
        """Get a summary of the output directory."""
        return f"\nAll outputs saved to: {self.output_dir}/"
