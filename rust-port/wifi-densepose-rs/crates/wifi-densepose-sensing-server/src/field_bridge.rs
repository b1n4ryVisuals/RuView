//! Bridge between sensing-server frame data and signal crate FieldModel
//! for eigenvalue-based person counting.
//!
//! The FieldModel decomposes CSI observations into environmental drift and
//! body perturbation via SVD eigenmodes. When calibrated, perturbation energy
//! provides a physics-grounded occupancy estimate that supplements the
//! score-based heuristic in `score_to_person_count`.

use std::collections::VecDeque;
use wifi_densepose_signal::ruvsense::field_model::{CalibrationStatus, FieldModel, FieldModelConfig};

use super::score_to_person_count;

/// Number of recent frames to feed into perturbation extraction.
const OCCUPANCY_WINDOW: usize = 50;

/// Perturbation energy threshold for detecting a second person.
const ENERGY_THRESH_2: f64 = 12.0;
/// Perturbation energy threshold for detecting a third person.
const ENERGY_THRESH_3: f64 = 25.0;

/// Create a FieldModelConfig for single-link mode (one ESP32 node = one link).
///
/// `min_calibration_frames` is lowered to 200 (~10 s at 20 Hz) so that a
/// short empty-room capture is enough to finalize the baseline.  The default
/// of 12,000 frames (10 min) is impractical for interactive calibration.
/// `n_subcarriers` is intentionally left at 0 here — the actual value is
/// patched in `maybe_feed_calibration` once the first real frame arrives, so
/// the model always matches the hardware's subcarrier count.
pub fn single_link_config() -> FieldModelConfig {
    FieldModelConfig {
        n_links: 1,
        min_calibration_frames: 200,
        ..FieldModelConfig::default()
    }
}

/// Estimate occupancy using the FieldModel when calibrated, falling back
/// to the score-based heuristic otherwise.
///
/// Prefers `estimate_occupancy()` (eigenvalue-based) when the model is
/// calibrated and enough frames are available. Falls back to perturbation
/// energy thresholds, then to the score heuristic.
pub fn occupancy_or_fallback(
    field: &FieldModel,
    frame_history: &VecDeque<Vec<f64>>,
    smoothed_score: f64,
    prev_count: usize,
) -> usize {
    match field.status() {
        CalibrationStatus::Fresh | CalibrationStatus::Stale => {
            let frames: Vec<Vec<f64>> = frame_history
                .iter()
                .rev()
                .take(OCCUPANCY_WINDOW)
                .cloned()
                .collect();

            if frames.is_empty() {
                return score_to_person_count(smoothed_score, prev_count);
            }

            // Try eigenvalue-based occupancy first (best accuracy).
            match field.estimate_occupancy(&frames) {
                Ok(count) => return count,
                Err(_) => {} // fall through to perturbation energy
            }

            // Fallback: perturbation energy thresholds.
            // FieldModel expects [n_links][n_subcarriers] — we use n_links=1.
            let observation = vec![frames[0].clone()];
            match field.extract_perturbation(&observation) {
                Ok(perturbation) => {
                    if perturbation.total_energy > ENERGY_THRESH_3 {
                        3
                    } else if perturbation.total_energy > ENERGY_THRESH_2 {
                        2
                    } else if perturbation.total_energy > 1.0 {
                        1
                    } else {
                        0
                    }
                }
                Err(_) => score_to_person_count(smoothed_score, prev_count),
            }
        }
        _ => score_to_person_count(smoothed_score, prev_count),
    }
}

/// Feed the latest frame to the FieldModel during calibration collection.
///
/// Takes `&mut Option<FieldModel>` so it can lazily recreate the model on the
/// very first frame with the actual subcarrier count reported by the hardware.
/// The default config uses `n_subcarriers: 56`, but ESP32-S3 typically sends
/// 114 subcarriers.  Without this fix every `feed_calibration` call silently
/// returns `DimensionMismatch` and the frame count stays at 0.
pub fn maybe_feed_calibration(field_opt: &mut Option<FieldModel>, frame_history: &VecDeque<Vec<f64>>) {
    let status = match field_opt.as_ref() {
        Some(f) => f.status(),
        None => return,
    };
    match status {
        CalibrationStatus::Uncalibrated | CalibrationStatus::Collecting => {}
        _ => return,
    }
    let latest = match frame_history.back() {
        Some(f) if !f.is_empty() => f,
        _ => return,
    };

    // On the very first frame (Uncalibrated, 0 frames collected), check whether
    // the configured n_subcarriers matches the live frame.  If not, recreate the
    // model with the correct size so that feed_calibration never returns
    // DimensionMismatch.
    if status == CalibrationStatus::Uncalibrated {
        let configured = field_opt.as_ref().unwrap().n_subcarriers();
        if configured != latest.len() {
            let new_cfg = FieldModelConfig {
                n_links: 1,
                n_subcarriers: latest.len(),
                min_calibration_frames: 200,
                ..FieldModelConfig::default()
            };
            match FieldModel::new(new_cfg) {
                Ok(new_fm) => { *field_opt = Some(new_fm); }
                Err(e) => {
                    tracing::warn!("FieldModel resize {configured}→{} failed: {e}", latest.len());
                    return;
                }
            }
        }
    }

    let field = field_opt.as_mut().unwrap();
    // Single-link observation: [1][n_subcarriers]
    let observations = vec![latest.clone()];
    if let Err(e) = field.feed_calibration(&observations) {
        tracing::debug!("FieldModel calibration feed: {e}");
    }
}

/// Parse node positions from a semicolon-delimited string.
///
/// Format: `"x,y,z;x,y,z;..."` where each coordinate is an `f32`.
/// Malformed entries are skipped with a warning log.
pub fn parse_node_positions(input: &str) -> Vec<[f32; 3]> {
    if input.is_empty() {
        return Vec::new();
    }
    input
        .split(';')
        .enumerate()
        .filter_map(|(idx, triplet)| {
            let parts: Vec<&str> = triplet.split(',').collect();
            if parts.len() != 3 {
                tracing::warn!("Skipping malformed node position entry {idx}: '{triplet}' (expected x,y,z)");
                return None;
            }
            match (parts[0].parse::<f32>(), parts[1].parse::<f32>(), parts[2].parse::<f32>()) {
                (Ok(x), Ok(y), Ok(z)) => Some([x, y, z]),
                _ => {
                    tracing::warn!("Skipping unparseable node position entry {idx}: '{triplet}'");
                    None
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node_positions() {
        let positions = parse_node_positions("0,0,1.5;3,0,1.5;1.5,3,1.5");
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], [0.0, 0.0, 1.5]);
        assert_eq!(positions[1], [3.0, 0.0, 1.5]);
        assert_eq!(positions[2], [1.5, 3.0, 1.5]);
    }

    #[test]
    fn test_parse_node_positions_empty() {
        let positions = parse_node_positions("");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_parse_node_positions_invalid() {
        let positions = parse_node_positions("abc;1,2,3");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_node_positions_partial_triplet() {
        let positions = parse_node_positions("1,2;3,4,5");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], [3.0, 4.0, 5.0]);
    }
}
