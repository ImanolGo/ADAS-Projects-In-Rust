//! GTSRB class names and what each sign means for speed.

/// The 43 GTSRB classes, in the order the model was trained on.
pub const GTSRB_LABELS: [&str; 43] = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing veh over 3.5 tons",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Veh > 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End speed + passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End no passing veh > 3.5 tons",
];

/// Human-readable name for a class id (or "unknown").
pub fn label(class_id: usize) -> &'static str {
    GTSRB_LABELS.get(class_id).copied().unwrap_or("unknown")
}

/// What a sign means for the speed you may drive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpeedMeaning {
    /// A new limit, in km/h.
    Limit(u32),
    /// "End of speed limit (80)": that specific limit no longer applies.
    EndOfLimit(u32),
    /// "End of all speed and passing limits".
    EndOfAllLimits,
}

/// Map a GTSRB class to its speed meaning. Every other sign is `None`.
///
/// The original Activity 3 treated class 6 ("end of 80") and class 32 ("end of
/// all limits") as "not a speed sign", so a car could never leave a limit.
pub fn speed_meaning(class_id: usize) -> Option<SpeedMeaning> {
    Some(match class_id {
        0 => SpeedMeaning::Limit(20),
        1 => SpeedMeaning::Limit(30),
        2 => SpeedMeaning::Limit(50),
        3 => SpeedMeaning::Limit(60),
        4 => SpeedMeaning::Limit(70),
        5 => SpeedMeaning::Limit(80),
        6 => SpeedMeaning::EndOfLimit(80),
        7 => SpeedMeaning::Limit(100),
        8 => SpeedMeaning::Limit(120),
        32 => SpeedMeaning::EndOfAllLimits,
        _ => return None,
    })
}

/// The class id of a speed-limit sign, if GTSRB has one for that value.
pub fn class_for_limit(kmh: u32) -> Option<usize> {
    (0..GTSRB_LABELS.len()).find(|&c| speed_meaning(c) == Some(SpeedMeaning::Limit(kmh)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_classes_round_trip() {
        for kmh in [20, 30, 50, 60, 70, 80, 100, 120] {
            let class = class_for_limit(kmh).unwrap();
            assert_eq!(speed_meaning(class), Some(SpeedMeaning::Limit(kmh)));
        }
        assert_eq!(class_for_limit(90), None);
        assert_eq!(speed_meaning(6), Some(SpeedMeaning::EndOfLimit(80)));
        assert_eq!(speed_meaning(14), None); // stop sign
        assert_eq!(label(99), "unknown");
    }
}
