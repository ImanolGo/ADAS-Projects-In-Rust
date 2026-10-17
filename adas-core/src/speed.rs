//! Speed-limit recognition over time: turn noisy per-frame sign predictions
//! into the limit the car should obey.
//!
//! A single frame is not enough: the CNN is ~100% "confident" even when it is
//! wrong. So a limit only changes when the same speed sign wins `votes` of the
//! last `window` observations *and* clearly beats the runner-up. If the camera
//! keeps flip-flopping between two readings (this model confuses 70 and 20),
//! the old limit stays: a missed sign is safer than a wrong one.

use std::collections::{HashMap, VecDeque};

use crate::classifier::Prediction;
use crate::labels::SpeedMeaning;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Ignore predictions below this softmax probability.
    pub min_confidence: f32,
    /// How many recent observations to vote over.
    pub window: usize,
    /// Votes needed (within `window`) to change the limit.
    pub votes: usize,
    /// The winner must lead the runner-up by at least this many votes.
    pub margin: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self { min_confidence: 0.6, window: 9, votes: 5, margin: 2 }
    }
}

/// What changed this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LimitEvent {
    /// A new limit is in force.
    Set(u32),
    /// No posted limit any more (the default for the road type applies).
    Cleared,
}

#[derive(Clone, Debug)]
pub struct SpeedLimitTracker {
    config: TrackerConfig,
    recent: VecDeque<Option<SpeedMeaning>>,
    current: Option<u32>,
}

impl SpeedLimitTracker {
    pub fn new(config: TrackerConfig) -> Self {
        assert!(config.votes >= 1 && config.votes <= config.window, "votes must be in 1..=window");
        Self { config, recent: VecDeque::with_capacity(config.window), current: None }
    }

    /// The limit currently in force, if any.
    pub fn current(&self) -> Option<u32> {
        self.current
    }

    /// Feed one frame: the sign prediction for this frame, or `None` if no sign was seen.
    pub fn observe(&mut self, prediction: Option<&Prediction>) -> Option<LimitEvent> {
        let meaning = prediction
            .filter(|p| p.confidence >= self.config.min_confidence)
            .and_then(Prediction::speed_meaning);
        self.observe_meaning(meaning)
    }

    /// Same as [`observe`](Self::observe), for an already-decoded meaning.
    pub fn observe_meaning(&mut self, meaning: Option<SpeedMeaning>) -> Option<LimitEvent> {
        if self.recent.len() == self.config.window {
            self.recent.pop_front();
        }
        self.recent.push_back(meaning);

        let winner = self.winner()?;
        let event = match winner {
            SpeedMeaning::Limit(kmh) if self.current != Some(kmh) => Some(LimitEvent::Set(kmh)),
            SpeedMeaning::EndOfLimit(_) | SpeedMeaning::EndOfAllLimits if self.current.is_some() => {
                Some(LimitEvent::Cleared)
            }
            _ => None,
        };
        if let Some(event) = event {
            self.current = match event {
                LimitEvent::Set(kmh) => Some(kmh),
                LimitEvent::Cleared => None,
            };
            // Start counting afresh, so one sign produces exactly one event.
            self.recent.clear();
        }
        event
    }

    fn winner(&self) -> Option<SpeedMeaning> {
        let mut counts: HashMap<SpeedMeaning, usize> = HashMap::new();
        for meaning in self.recent.iter().flatten() {
            *counts.entry(*meaning).or_default() += 1;
        }
        let mut ranked: Vec<(SpeedMeaning, usize)> = counts.into_iter().collect();
        ranked.sort_by_key(|r| std::cmp::Reverse(r.1));
        let (best, votes) = *ranked.first()?;
        let runner_up = ranked.get(1).map_or(0, |r| r.1);
        (votes >= self.config.votes && votes >= runner_up + self.config.margin).then_some(best)
    }
}

impl Default for SpeedLimitTracker {
    fn default() -> Self {
        Self::new(TrackerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::prediction_from_scores;
    use crate::labels::class_for_limit;

    fn seen(class_id: usize, confidence: f32) -> Prediction {
        let mut scores = vec![0.0; 43];
        scores[class_id] = confidence;
        prediction_from_scores(&scores).unwrap()
    }

    fn tracker() -> SpeedLimitTracker {
        SpeedLimitTracker::new(TrackerConfig { min_confidence: 0.6, window: 5, votes: 3, margin: 2 })
    }

    #[test]
    fn a_flip_flopping_reading_keeps_the_old_limit() {
        // The real model reads some 70 signs as 20 about half the time.
        let mut t =
            SpeedLimitTracker::new(TrackerConfig { min_confidence: 0.6, window: 9, votes: 5, margin: 2 });
        for _ in 0..5 {
            t.observe_meaning(Some(SpeedMeaning::Limit(30)));
        }
        assert_eq!(t.current(), Some(30));
        for kmh in [70, 20, 20, 70, 20, 70, 70, 20, 20, 70] {
            assert_eq!(t.observe_meaning(Some(SpeedMeaning::Limit(kmh))), None, "changed on a coin flip");
        }
        assert_eq!(t.current(), Some(30));
    }

    #[test]
    fn a_limit_needs_several_frames() {
        let mut t = tracker();
        let p70 = seen(class_for_limit(70).unwrap(), 0.99);
        assert_eq!(t.observe(Some(&p70)), None);
        assert_eq!(t.observe(Some(&p70)), None);
        assert_eq!(t.observe(Some(&p70)), Some(LimitEvent::Set(70)));
        assert_eq!(t.current(), Some(70));
        // More frames of the same sign do not fire again.
        assert_eq!(t.observe(Some(&p70)), None);
        assert_eq!(t.observe(Some(&p70)), None);
        assert_eq!(t.observe(Some(&p70)), None);
    }

    #[test]
    fn single_glitches_are_ignored() {
        let mut t = tracker();
        let p30 = seen(class_for_limit(30).unwrap(), 0.99);
        let p120 = seen(class_for_limit(120).unwrap(), 1.0);
        for p in [&p30, &p120, &p30, &p30] {
            t.observe(Some(p));
        }
        assert_eq!(t.current(), Some(30));
        assert_eq!(t.observe(Some(&p120)), None); // one stray frame cannot change it
    }

    #[test]
    fn low_confidence_and_other_signs_do_not_vote() {
        let mut t = tracker();
        let weak = seen(class_for_limit(50).unwrap(), 0.3);
        let stop = seen(14, 1.0);
        for _ in 0..10 {
            assert_eq!(t.observe(Some(&weak)), None);
            assert_eq!(t.observe(Some(&stop)), None);
            assert_eq!(t.observe(None), None);
        }
        assert_eq!(t.current(), None);
    }

    #[test]
    fn end_signs_clear_the_limit() {
        let mut t = tracker();
        for _ in 0..3 {
            t.observe_meaning(Some(SpeedMeaning::Limit(80)));
        }
        assert_eq!(t.current(), Some(80));
        let mut events = vec![];
        for _ in 0..3 {
            events.extend(t.observe(Some(&seen(6, 0.95))));
        }
        assert_eq!(events, vec![LimitEvent::Cleared]);
        assert_eq!(t.current(), None);
    }

    #[test]
    fn a_new_limit_replaces_the_old_one() {
        let mut t = tracker();
        for kmh in [30, 30, 30, 70, 70, 70] {
            t.observe_meaning(Some(SpeedMeaning::Limit(kmh)));
        }
        assert_eq!(t.current(), Some(70));
    }
}
