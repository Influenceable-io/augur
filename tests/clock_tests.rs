use augur::clock::Clock;
use chrono::{Duration, Utc};

#[test]
fn test_clock_time_step() {
    let mut clock = Clock::new(1);
    assert_eq!(clock.get_time_step(), "0");

    clock.time_step += 1;
    assert_eq!(clock.get_time_step(), "1");
}

#[test]
fn test_clock_time_transfer_acceleration() {
    let clock = Clock::new(10);
    let start_time = Utc::now();

    // Simulate 1 second of real elapsed time
    let fake_now = clock.real_start_time + Duration::seconds(1);
    let sim_time = clock.time_transfer(fake_now, start_time);

    // With k=10, 1 real second should map to 10 simulated seconds
    let expected = start_time + Duration::seconds(10);
    let diff = (sim_time - expected).num_milliseconds().abs();
    assert!(
        diff < 50,
        "Expected sim_time ~{expected}, got {sim_time} (diff {diff}ms)"
    );
}

#[test]
fn test_clock_k1_realtime() {
    let clock = Clock::new(1);
    let start_time = Utc::now();

    // With k=1, sim time should approximately equal real elapsed time
    let fake_now = clock.real_start_time + Duration::seconds(5);
    let sim_time = clock.time_transfer(fake_now, start_time);

    let expected = start_time + Duration::seconds(5);
    let diff = (sim_time - expected).num_milliseconds().abs();
    assert!(
        diff < 50,
        "k=1 should keep sim_time ~= real time; diff was {diff}ms"
    );
}
