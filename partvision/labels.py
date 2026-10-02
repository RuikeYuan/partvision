"""Label taxonomy helpers.

Many car parts come in left/right pairs that are mirror images of each other
(headlights, mirrors, taillights, doors). A standard horizontal-flip augmentation
would silently turn a left headlight into a right one while keeping the label
"left", which teaches the model to ignore exactly the feature that matters.

Convention: sided labels end with ``_left`` / ``_right``. Everything else is
treated as symmetric (a wheel rim flipped is still a wheel rim).
"""

from __future__ import annotations

import re

LEFT, RIGHT = "_left", "_right"
_LABEL_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def is_valid_label(label: str) -> bool:
    return bool(_LABEL_RE.match(label)) and len(label) <= 64


def is_sided(label: str) -> bool:
    return label.endswith(LEFT) or label.endswith(RIGHT)


def mirror_label(label: str) -> str:
    """Label of the part you get when you mirror the image."""
    if label.endswith(LEFT):
        return label[: -len(LEFT)] + RIGHT
    if label.endswith(RIGHT):
        return label[: -len(RIGHT)] + LEFT
    return label


def build_flip_map(classes: list[str]) -> list[int]:
    """For each class index, the class index after a horizontal flip.

    -1 means "do not flip": the class is sided but its mirror class is not in the
    label set, so a flipped image would have no correct label.
    """
    index = {c: i for i, c in enumerate(classes)}
    return [index.get(mirror_label(c), -1) for c in classes]
