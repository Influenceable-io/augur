use chrono::{DateTime, Utc};

/// Time simulation with k-factor acceleration.
///
/// Mirrors OASIS's Clock class:
/// - `k=1`: real-time
/// - `k=2`: 2x speed (2 hours simulated per 1 hour real)
/// - `time_step`: discrete step counter (used by Twitter mode)
/// - `time_transfer`: continuous time mapping (used by Reddit mode)
pub struct Clock {
    /// Time acceleration factor.
    pub k: u64,
    /// Real-world start time of the simulation.
    pub real_start_time: DateTime<Utc>,
    /// Discrete time step counter.
    pub time_step: u64,
}

impl Clock {
    pub fn new(k: u64) -> Self {
        Self {
            k,
            real_start_time: Utc::now(),
            time_step: 0,
        }
    }

    /// Map real elapsed time to accelerated simulated time.
    ///
    /// `start_time` is the simulated epoch (when the simulation "begins" in sim-time).
    pub fn time_transfer(&self, now: DateTime<Utc>, start_time: DateTime<Utc>) -> DateTime<Utc> {
        let elapsed = now - self.real_start_time;
        let accelerated = elapsed * self.k as i32;
        start_time + accelerated
    }

    /// Return the current time step as a string (matches OASIS).
    pub fn get_time_step(&self) -> String {
        self.time_step.to_string()
    }
}
