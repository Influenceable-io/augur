use rand::seq::SliceRandom;

use super::{PostRow, UserRow};

/// Random recommendation: distribute random posts to all users.
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    let mut rng = rand::rng();
    let mut recommendations = Vec::new();

    let post_ids: Vec<i64> = post_table.iter().map(|p| p.post_id).collect();

    for user in user_table {
        let mut shuffled = post_ids.clone();
        shuffled.shuffle(&mut rng);
        let count = max_rec_post_len.min(shuffled.len());
        for &post_id in &shuffled[..count] {
            recommendations.push((user.user_id, post_id));
        }
    }

    recommendations
}
