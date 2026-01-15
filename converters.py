"""Convert Perceptron SDK types to FiftyOne types.

This module provides conversion functions between Perceptron SDK geometry types
(SinglePoint, BoundingBox, Polygon) and FiftyOne annotation types (Detection,
Keypoint, Polyline).

Both systems use a 0-1000 normalized coordinate system, but FiftyOne requires
coordinates normalized to [0, 1].
"""
import logging
import fiftyone as fo
from typing import List, Optional
from perceptron import BoundingBox, SinglePoint, Polygon, pt

logger = logging.getLogger(__name__)

# Perceptron normalized coordinate max
COORD_MAX = 1000.0


def perceptron_to_fiftyone_detection(
    box: BoundingBox, is_ocr: bool = False
) -> fo.Detection:
    """Convert BoundingBox to fo.Detection with [0,1] normalized coords.

    Args:
        box: Perceptron BoundingBox with top_left and bottom_right points
        is_ocr: If True, also stores the label as a 'text' attribute

    Returns:
        FiftyOne Detection with normalized bounding_box [x, y, w, h]
    """
    x1 = box.top_left.x / COORD_MAX
    y1 = box.top_left.y / COORD_MAX
    x2 = box.bottom_right.x / COORD_MAX
    y2 = box.bottom_right.y / COORD_MAX

    label = box.mention or "object"
    detection = fo.Detection(
        label=label,
        bounding_box=[x1, y1, x2 - x1, y2 - y1]
    )

    if is_ocr:
        detection["text"] = label

    return detection


def perceptron_to_fiftyone_keypoint(point: SinglePoint) -> tuple:
    """Convert SinglePoint to (x, y) tuple with [0,1] normalized coords.

    Args:
        point: Perceptron SinglePoint with x, y coordinates

    Returns:
        Tuple of (x, y) normalized to [0, 1]
    """
    return (point.x / COORD_MAX, point.y / COORD_MAX)


def perceptron_to_fiftyone_polyline(
    polygon: Polygon, is_ocr: bool = False
) -> fo.Polyline:
    """Convert Polygon to fo.Polyline with [0,1] normalized coords.

    Args:
        polygon: Perceptron Polygon with hull of SinglePoint vertices
        is_ocr: If True, also stores the label as a 'text' attribute

    Returns:
        FiftyOne Polyline with normalized points
    """
    points = [(p.x / COORD_MAX, p.y / COORD_MAX) for p in polygon.hull]
    label = polygon.mention or "polygon"

    polyline = fo.Polyline(
        label=label,
        points=[points],
        closed=True,
        filled=True
    )

    if is_ocr:
        polyline["text"] = label

    return polyline


def box_to_center_point(box: BoundingBox) -> SinglePoint:
    """Convert BoundingBox to center SinglePoint.

    Used when model returns boxes but points were requested.

    Args:
        box: Perceptron BoundingBox

    Returns:
        SinglePoint at the center of the box
    """
    cx = (box.top_left.x + box.bottom_right.x) // 2
    cy = (box.top_left.y + box.bottom_right.y) // 2
    return pt(cx, cy, mention=box.mention)


# Batch converters

def perceptron_to_fiftyone_detections(
    boxes: List[BoundingBox], is_ocr: bool = False
) -> fo.Detections:
    """Convert list of BoundingBox to fo.Detections.

    Args:
        boxes: List of Perceptron BoundingBox objects
        is_ocr: If True, stores labels as 'text' attributes

    Returns:
        FiftyOne Detections container
    """
    if not boxes:
        return fo.Detections(detections=[])

    detections = []
    for b in boxes:
        try:
            detections.append(perceptron_to_fiftyone_detection(b, is_ocr))
        except Exception as e:
            logger.debug(f"Error processing box {b}: {e}")
            continue

    return fo.Detections(detections=detections)


def perceptron_to_fiftyone_keypoints(points: List[SinglePoint]) -> fo.Keypoints:
    """Convert list of SinglePoint to fo.Keypoints.

    Groups points by label for FiftyOne's Keypoints format.

    Args:
        points: List of Perceptron SinglePoint objects

    Returns:
        FiftyOne Keypoints container
    """
    if not points:
        return fo.Keypoints(keypoints=[])

    # Group points by label
    by_label = {}
    for p in points:
        try:
            label = p.mention or "point"
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(perceptron_to_fiftyone_keypoint(p))
        except Exception as e:
            logger.debug(f"Error processing point {p}: {e}")
            continue

    return fo.Keypoints(
        keypoints=[fo.Keypoint(label=k, points=v) for k, v in by_label.items()]
    )


def perceptron_to_fiftyone_polylines(
    polygons: List[Polygon], is_ocr: bool = False
) -> fo.Polylines:
    """Convert list of Polygon to fo.Polylines.

    Args:
        polygons: List of Perceptron Polygon objects
        is_ocr: If True, stores labels as 'text' attributes

    Returns:
        FiftyOne Polylines container
    """
    if not polygons:
        return fo.Polylines(polylines=[])

    polylines = []
    for p in polygons:
        try:
            # Skip polygons with fewer than 3 vertices
            if not p.hull or len(p.hull) < 3:
                continue
            polylines.append(perceptron_to_fiftyone_polyline(p, is_ocr))
        except Exception as e:
            logger.debug(f"Error processing polygon {p}: {e}")
            continue

    return fo.Polylines(polylines=polylines)
