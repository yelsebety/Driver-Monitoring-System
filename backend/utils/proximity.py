"""Where is the driver's mouth and face? Used to ignore objects that are in the
frame but not actually being used (e.g. a bottle standing on the dashboard).

All boxes are (x1, y1, x2, y2) in pixels, the same format YOLO detections use.
"""

# MediaPipe face-mesh landmarks: 13 = upper inner lip, 14 = lower inner lip
UPPER_LIP, LOWER_LIP = 13, 14


def compute_regions(points, mouth_half_width=0.35, mouth_half_height=0.30,
                    face_side_pad=0.40, face_vert_pad=0.20):
    """points: the (N, 2) array from FaceMesh.process(), or None if no face.

    Returns {"mouth": box or None, "face": box or None}. All sizes are
    fractions of the face width, so they work at any distance from the camera.
      mouth - the area around the lips (for eating / drinking)
      face  - the face plus some room on each side for the ear (for phone use)
    """
    if points is None:
        return {"mouth": None, "face": None}

    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    fw = float(x2 - x1)

    mx, my = points[[UPPER_LIP, LOWER_LIP]].mean(axis=0)
    mouth = (mx - mouth_half_width * fw, my - mouth_half_height * fw,
             mx + mouth_half_width * fw, my + mouth_half_height * fw)
    face = (x1 - face_side_pad * fw, y1 - face_vert_pad * fw,
            x2 + face_side_pad * fw, y2 + face_vert_pad * fw)
    return {"mouth": tuple(float(v) for v in mouth),
            "face": tuple(float(v) for v in face)}


def boxes_overlap(a, b):
    """True if two (x1, y1, x2, y2) boxes share any area."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
