//! The 80 COCO class names the YOLOX detector was trained on, plus the subset
//! that matters for driving.

/// Class names in the order YOLOX's COCO head emits them.
pub const COCO_LABELS: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Number of classes the COCO head outputs.
pub const NUM_CLASSES: usize = COCO_LABELS.len();

/// COCO class ids worth showing a driver: vehicles, people, and traffic signs.
pub const ADAS_CLASSES: [usize; 9] = [0, 1, 2, 3, 5, 6, 7, 9, 11];

/// The COCO "traffic light" id.
pub const TRAFFIC_LIGHT: usize = 9;

/// Human-readable name for a class id (or "unknown").
pub fn label(class_id: usize) -> &'static str {
    COCO_LABELS.get(class_id).copied().unwrap_or("unknown")
}

/// Class id for an exact name, e.g. `id("car")`.
pub fn id(name: &str) -> Option<usize> {
    COCO_LABELS.iter().position(|&n| n == name)
}

/// Is this class one of the driving-relevant ones?
pub fn is_adas(class_id: usize) -> bool {
    ADAS_CLASSES.contains(&class_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_the_canonical_coco_order() {
        assert_eq!(label(0), "person");
        assert_eq!(label(2), "car");
        assert_eq!(label(9), "traffic light");
        assert_eq!(id("traffic light"), Some(TRAFFIC_LIGHT));
        assert_eq!(id("car"), Some(2));
        assert_eq!(label(999), "unknown");
        assert_eq!(id("nonsense"), None);
    }
}
