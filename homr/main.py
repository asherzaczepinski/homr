import argparse
import glob
import os
import sys
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort

from homr import color_adjust, download_utils
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.deskew import deskew_image
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions, MultiStaff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.config import segnet_path_onnx, segnet_path_onnx_fp16
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_staff_positions, save_staff_positions
from homr.title_detection import detect_title, download_ocr_weights
from homr.transformer.configs import Config, default_config
from homr.type_definitions import NDArray
from homr.visualization_output import VisualizationOutput

# Import Orchestra-AI-2 for accidental detection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Orchestra-AI-2'))
try:
    from test_accidentals import AccidentalDetector
    ACCIDENTAL_DETECTOR_AVAILABLE = True
except ImportError as e:
    ACCIDENTAL_DETECTOR_AVAILABLE = False
    eprint("⚠ WARNING: Orchestra-AI-2 accidental detector not available")
    eprint(f"⚠ Import error: {e}")
    eprint("⚠ Will use geometric fallback method for accidental detection")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class AccidentalDetection:
    """Represents a detected accidental with its class name."""
    bbox: RotatedBoundingBox
    class_name: str
    confidence: float


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
        accidentals: list[AccidentalDetection] | None = None,
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines
        self.accidentals = accidentals or []


class InvalidProgramArgumentException(Exception):
    """Raise this exception for issues which the user can address."""


class GpuSupport(Enum):
    No = "no"
    AUTO = "auto"
    FORCE = "force"


def get_predictions(
    original: NDArray,
    preprocessed: NDArray,
    img_path: str,
    enable_cache: bool,
    use_gpu_inference: bool,
) -> InputPredictions:
    result = extract(
        preprocessed,
        img_path,
        step_size=320,
        use_cache=enable_cache,
        use_gpu_inference=use_gpu_inference,
    )
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool, use_gpu_inference: bool
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidProgramArgumentException(
            "The file format is not supported, please provide a JPG or PNG image file:" + image_path
        )

    # Apply deskewing before other preprocessing
    eprint("Deskewing image...")
    image = deskew_image(image)

    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _background = color_adjust.color_adjust(image, 40)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache, use_gpu_inference)
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def calculate_overlap_percentage(box1, box2):
    """Calculate what percentage of box1 overlaps with box2."""
    # Get bounding rectangles
    x1_min, y1_min = box1.top_left
    x1_max, y1_max = box1.bottom_right
    x2_min, y2_min = box2.top_left
    x2_max, y2_max = box2.bottom_right

    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = x_overlap * y_overlap

    # Calculate box1's area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)

    if box1_area == 0:
        return 0

    return overlap_area / box1_area


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    # Use Orchestra-AI-2 for accidental detection
    accidental_detections = []
    clefs_keys = []  # Will store bounding boxes for compatibility
    use_orchestra_ai = ACCIDENTAL_DETECTOR_AVAILABLE

    if use_orchestra_ai:
        eprint("Using Orchestra-AI-2 for accidental detection")
        try:
            # Initialize Orchestra-AI-2 accidental detector
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Orchestra-AI-2', 'weights', 'best.pt')
            detector = AccidentalDetector(model_path=model_path, confidence_threshold=0.3)

            # Convert preprocessed image to RGB for detection
            img_rgb = cv2.cvtColor(predictions.preprocessed, cv2.COLOR_GRAY2RGB)

            # Run detection
            detections = detector.detect_in_image(img_rgb)
            eprint(f"Found {len(detections)} accidentals using Orchestra-AI-2")

            # Remove overlapping detections (>50% overlap), keeping higher confidence
            filtered_detections = []
            for i, det in enumerate(detections):
                should_keep = True
                for j, other_det in enumerate(detections):
                    if i == j:
                        continue

                    # Calculate overlap percentage
                    x_overlap = max(0, min(det.x2, other_det.x2) - max(det.x1, other_det.x1))
                    y_overlap = max(0, min(det.y2, other_det.y2) - max(det.y1, other_det.y1))
                    overlap_area = x_overlap * y_overlap

                    det_area = (det.x2 - det.x1) * (det.y2 - det.y1)
                    if det_area == 0:
                        continue

                    overlap_pct = overlap_area / det_area

                    # If overlap > 50% and this detection has lower confidence, remove it
                    if overlap_pct > 0.5 and det.confidence < other_det.confidence:
                        should_keep = False
                        break

                if should_keep:
                    filtered_detections.append(det)

            removed_count = len(detections) - len(filtered_detections)
            if removed_count > 0:
                eprint(f"Removed {removed_count} overlapping accidentals (>50% overlap, lower confidence)")

            detections = filtered_detections
            eprint(f"Final count: {len(detections)} accidentals after overlap removal")

            # Convert Orchestra-AI-2 detections to our format
            for det in detections:
                # Create a simple bounding box from the detection
                center = ((det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2)
                size = (det.x2 - det.x1, det.y2 - det.y1)

                # Create a simple rectangular contour from the bounding box
                x1, y1 = int(det.x1), int(det.y1)
                x2, y2 = int(det.x2), int(det.y2)
                contours = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

                # Create RotatedBoundingBox in OpenCV RotatedRect format: ((center_x, center_y), (width, height), angle)
                rotated_rect = (center, size, 0.0)
                bbox = RotatedBoundingBox(rotated_rect, contours)

                accidental_detections.append(
                    AccidentalDetection(
                        bbox=bbox,
                        class_name=det.class_name,
                        confidence=det.confidence
                    )
                )
                clefs_keys.append(bbox)  # For compatibility with existing code

            # Count by type
            type_counts = {}
            for det in accidental_detections:
                type_counts[det.class_name] = type_counts.get(det.class_name, 0) + 1

            eprint("Accidental breakdown:")
            for class_name, count in sorted(type_counts.items()):
                eprint(f"  {class_name}: {count}")

        except Exception as e:
            eprint(f"⚠ ERROR using Orchestra-AI-2 detector: {e}")
            eprint("⚠ FALLING BACK to geometric method")
            use_orchestra_ai = False

    # Fallback to original method if Orchestra-AI-2 is not available
    if not use_orchestra_ai or len(accidental_detections) == 0:
        eprint("⚠ FALLBACK: Creating bounds for accidentals (sharps, flats, naturals) - using geometric method")
        # Detect all symbols first
        all_symbols = create_rotated_bounding_boxes(
            predictions.clefs_keys, min_size=(10, 15), max_size=(1000, 1000), skip_merging=True
        )

        # Separate accidentals from clefs based on size
        # Accidentals are typically 15-35 pixels tall, clefs are 40+ pixels tall
        accidentals_only = []
        clefs_only = []
        for symbol in all_symbols:
            # Get symbol height
            height = symbol.size[1]
            if height < 38:
                accidentals_only.append(symbol)
            else:
                clefs_only.append(symbol)

        eprint(f"Found {len(all_symbols)} symbols: {len(accidentals_only)} accidentals, {len(clefs_only)} clefs")

        # Filter out accidentals that overlap >50% with clefs
        filtered_accidentals = []
        for accidental in accidentals_only:
            overlaps_clef = False
            for clef in clefs_only:
                overlap_pct = calculate_overlap_percentage(accidental, clef)
                if overlap_pct > 0.5:
                    overlaps_clef = True
                    break

            if not overlaps_clef:
                filtered_accidentals.append(accidental)

        removed_overlap = len(accidentals_only) - len(filtered_accidentals)
        eprint(f"Removed {removed_overlap} accidentals overlapping with clefs (>50%)")
        eprint(f"Final: {len(filtered_accidentals)} accidentals")

        clefs_keys = filtered_accidentals

    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines, accidental_detections)


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    write_staff_positions: bool
    read_staff_positions: bool
    selected_staff: int
    use_gpu_inference: bool


def process_image(
    image_path: str,
    config: ProcessingConfig,
    xml_generator_args: XmlGeneratorArguments,
) -> None:
    eprint("Processing " + image_path)

    # Create visualization output manager (always enabled)
    viz_output = VisualizationOutput(image_path)

    xml_file = replace_extension(image_path, ".musicxml")
    debug_cleanup: Debug | None = None
    try:
        if config.read_staff_positions:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read " + image_path)
            image = resize_image(image)
            debug = Debug(image, image_path, config.enable_debug)
            staff_position_files = replace_extension(image_path, ".txt")
            multi_staffs = load_staff_positions(
                debug, image, staff_position_files, config.selected_staff
            )
            title = ""
        else:
            multi_staffs, image, debug, title_future = detect_staffs_in_image(image_path, config, viz_output)
        debug_cleanup = debug

        # DETECTION ONLY MODE: Skip all parsing and XML generation
        # All visualizations have already been saved by detect_staffs_in_image
        eprint("Detection complete! Skipping parsing and XML generation.")
        eprint(viz_output.get_summary())
        return

        # The code below is now skipped (parsing and XML generation)
        # transformer_config = Config()
        # transformer_config.use_gpu_inference = config.use_gpu_inference
        #
        # result_staffs = parse_staffs(
        #     debug,
        #     multi_staffs,
        #     image,
        #     selected_staff=config.selected_staff,
        #     config=transformer_config,
        # )
        #
        # title = title_future.result(60)
        # eprint("Found title:", title)
        #
        # eprint("Writing XML", result_staffs)
        # xml = generate_xml(xml_generator_args, result_staffs, title)
        #
        # # Save XML to output folder
        # import tempfile
        # with tempfile.NamedTemporaryFile(mode='w', suffix='.musicxml', delete=False) as tmp:
        #     xml.write(tmp.name)
        #     with open(tmp.name, 'r') as f:
        #         xml_string = f.read()
        # import os as os_temp
        # os_temp.remove(tmp.name)
        # xml_file = viz_output.save_musicxml(xml_string)
        #
        # eprint("Finished parsing " + str(len(result_staffs)) + " staves")
        teaser_file = replace_extension(image_path, "_teaser.png")
        if config.write_staff_positions:
            staff_position_files = replace_extension(image_path, ".txt")
            save_staff_positions(multi_staffs, image.shape, staff_position_files)
        debug.write_teaser(teaser_file, multi_staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint(viz_output.get_summary())
    except:
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        if debug_cleanup is not None:
            debug_cleanup.clean_debug_files_from_previous_runs()


def detect_staffs_in_image(
    image_path: str, config: ProcessingConfig, viz_output: VisualizationOutput | None = None
) -> tuple[list[MultiStaff], NDArray, Debug, Future[str]]:
    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.use_gpu_inference
    )

    # Set original image for visualization output
    if viz_output is not None:
        viz_output.set_original_image(predictions.original)

    symbols = predict_symbols(debug, predictions)

    # Save symbols detection visualization
    if viz_output is not None:
        viz_output.save_symbols_detection(symbols.clefs_keys, symbols.accidentals)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    # Calculate average staff line height (thickness) from staff fragments
    if len(symbols.staff_fragments) > 0:
        staff_line_heights = [fragment.size[1] for fragment in symbols.staff_fragments]
        average_staff_line_height = float(np.median(staff_line_heights))
        eprint(f"Average staff line height (thickness): {average_staff_line_height:.2f} pixels")
    else:
        average_staff_line_height = 2.0  # fallback
        eprint("Warning: No staff fragments found, using fallback line height")

    # Calculate average distance between staff lines (spacing)
    # Sort staff fragments by Y position and calculate distances between consecutive lines
    if len(symbols.staff_fragments) > 5:
        # Get Y centers of all staff line fragments
        staff_y_positions = sorted([fragment.center[1] for fragment in symbols.staff_fragments])

        # Calculate distances between consecutive lines
        line_distances = []
        for i in range(len(staff_y_positions) - 1):
            distance = staff_y_positions[i + 1] - staff_y_positions[i]
            # Only consider reasonable distances (filter out lines from same staff vs different staff)
            if 5 < distance < 50:  # reasonable range for lines in same staff
                line_distances.append(distance)

        if line_distances:
            average_line_spacing = float(np.median(line_distances))
            eprint(f"Average distance between staff lines (spacing): {average_line_spacing:.2f} pixels")
        else:
            average_line_spacing = 16.0  # fallback
            eprint("Warning: Could not calculate line spacing, using fallback")
    else:
        average_line_spacing = 16.0  # fallback
        eprint("Warning: Not enough staff fragments, using fallback spacing")

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads (before filtering)")

    # Minimum note size = average spacing between lines (the difference)
    min_note_size = average_line_spacing
    eprint(f"Minimum note size threshold: {min_note_size:.2f}px (average spacing between lines)")

    filtered_noteheads = []
    for notehead in noteheads_with_stems:
        width = notehead.notehead.size[0]
        height = notehead.notehead.size[1]
        if width >= min_note_size and height >= min_note_size:
            filtered_noteheads.append(notehead)

    removed_count = len(noteheads_with_stems) - len(filtered_noteheads)
    eprint(f"Filtered notes: kept {len(filtered_noteheads)}, removed {removed_count} (threshold: {min_note_size:.2f}px)")
    noteheads_with_stems = filtered_noteheads

    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found after filtering")

    # Save notes detection visualization
    if viz_output is not None:
        viz_output.save_notes_detection(noteheads_with_stems)

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    # Save measures (bar lines) detection visualization
    if viz_output is not None:
        viz_output.save_measures_detection(bar_line_boxes)

    debug.write_bounding_boxes(
        "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
    )
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    if len(staffs) == 0:
        raise Exception("No staffs found")
    title_future = detect_title(debug, staffs[0])
    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    # Save staff detection visualization
    if viz_output is not None:
        viz_output.save_staff_detection(staffs)
        # Save note positions grid visualization
        viz_output.save_note_positions_grid(staffs)
        # Save note splitting analysis (shows which line each note in a chord belongs to)
        viz_output.save_note_splitting_analysis(staffs, noteheads_with_stems)
        # Save accidental positions analysis (shows which red line each accidental belongs to)
        viz_output.save_accidental_positions_analysis(staffs, symbols.accidentals)
        # Save accidental effects visualization (circles notes affected by sharps/flats)
        viz_output.save_accidental_affected_notes(staffs, symbols.accidentals, noteheads_with_stems)

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    debug.write_threshold_image("brace_dot", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors("notes", multi_staffs, notes)

    return multi_staffs, predictions.preprocessed, debug, title_future


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights(use_gpu_inference: bool) -> None:
    base_url = "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/"
    if use_gpu_inference:
        models = [
            segnet_path_onnx_fp16,
            default_config.filepaths.encoder_path_fp16,
            default_config.filepaths.decoder_path_fp16,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]
    else:
        models = [
            segnet_path_onnx,
            default_config.filepaths.encoder_path,
            default_config.filepaths.decoder_path,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]

    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model):
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)

                destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homer", description="An optical music recognition (OMR) system"
    )
    parser.add_argument(
        "image", type=str, nargs="?", default="input.png",
        help="Path to the image to process (default: input.png)"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Downloads the models if they are missing and then exits. "
        + "You don't have to call init before processing images, "
        + "it's only useful if you want to prepare for example a Docker image.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true", help="Read an existing cache file or create a new one"
    )
    parser.add_argument(
        "--output-large-page",
        action="store_true",
        help="Adds instructions to the musicxml so that it gets rendered on larger pages",
    )
    parser.add_argument(
        "--output-metronome", type=int, help="Adds a metronome to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--output-tempo", type=int, help="Adds a tempo to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--write-staff-positions",
        action="store_true",
        help="Writes the position of all detected staffs to a txt file.",
    )
    parser.add_argument(
        "--read-staff-positions",
        action="store_true",
        help="Reads the position of all staffs from a txt file instead"
        + " of running the built-in staff detection.",
    )
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()

    use_gpu_inference = (
        args.gpu == GpuSupport.AUTO and has_gpu_support
    ) or args.gpu == GpuSupport.FORCE

    download_weights(use_gpu_inference)
    if args.init:
        download_ocr_weights()
        eprint("Init finished")
        return

    config = ProcessingConfig(
        args.debug,
        args.cache,
        args.write_staff_positions,
        args.read_staff_positions,
        -1,
        use_gpu_inference,
    )

    xml_generator_args = XmlGeneratorArguments(
        args.output_large_page, args.output_metronome, args.output_tempo
    )

    # Check if input file exists
    if not os.path.isfile(args.image):
        eprint(f"Error: Input file '{args.image}' not found")
        sys.exit(1)

    try:
        process_image(args.image, config, xml_generator_args)
    except InvalidProgramArgumentException as e:
        eprint(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
