use std::collections::HashSet;

use augur::recsys::{random, reddit, twitter, twhin, PostRow, TraceRow, UserRow};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_users(n: usize) -> Vec<UserRow> {
    (0..n as i64)
        .map(|i| UserRow {
            user_id: i + 1,
            bio: format!("User {}", i),
        })
        .collect()
}

fn make_posts(n: usize) -> Vec<PostRow> {
    (0..n as i64)
        .map(|i| PostRow {
            post_id: i + 1,
            user_id: (i % 3) + 1,
            content: format!("Post {}", i),
            created_at: "2026-04-01 12:00:00".to_string(),
            num_likes: i * 2,
            num_dislikes: i / 2,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Random recsys
// ---------------------------------------------------------------------------

#[test]
fn test_random_recsys() {
    let users = make_users(3);
    let posts = make_posts(10);
    let recs = random::recommend(&users, &posts, 5);

    // Each user should get exactly 5 recommendations
    for user in &users {
        let user_recs: Vec<_> = recs.iter().filter(|(uid, _)| *uid == user.user_id).collect();
        assert_eq!(
            user_recs.len(),
            5,
            "user {} should get 5 recs, got {}",
            user.user_id,
            user_recs.len()
        );
    }
}

#[test]
fn test_random_recsys_max_capped() {
    let users = make_users(2);
    let posts = make_posts(3);
    let recs = random::recommend(&users, &posts, 100);

    // max requested is 100, but only 3 posts exist → each user gets 3
    for user in &users {
        let user_recs: Vec<_> = recs.iter().filter(|(uid, _)| *uid == user.user_id).collect();
        assert_eq!(
            user_recs.len(),
            3,
            "user {} should get 3 recs (capped at post count), got {}",
            user.user_id,
            user_recs.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Reddit hot-score
// ---------------------------------------------------------------------------

#[test]
fn test_reddit_hot_score_ordering() {
    let users = make_users(1);
    let posts = vec![
        PostRow {
            post_id: 1,
            user_id: 99,
            content: "low".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 5,
            num_dislikes: 0,
        },
        PostRow {
            post_id: 2,
            user_id: 99,
            content: "mid".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 50,
            num_dislikes: 0,
        },
        PostRow {
            post_id: 3,
            user_id: 99,
            content: "high".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 500,
            num_dislikes: 0,
        },
    ];

    let recs = reddit::recommend(&users, &posts, 3);
    let post_ids: Vec<i64> = recs.iter().map(|(_, pid)| *pid).collect();

    // Highest likes first
    assert_eq!(post_ids, vec![3, 2, 1], "posts should be ordered by likes descending");
}

#[test]
fn test_reddit_negative_score() {
    let users = make_users(1);
    let posts = vec![
        PostRow {
            post_id: 1,
            user_id: 99,
            content: "positive".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 50,
            num_dislikes: 0,
        },
        PostRow {
            post_id: 2,
            user_id: 99,
            content: "negative".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 0,
            num_dislikes: 50,
        },
    ];

    let recs = reddit::recommend(&users, &posts, 2);
    let post_ids: Vec<i64> = recs.iter().map(|(_, pid)| *pid).collect();

    // Positive-scored post should rank higher
    assert_eq!(
        post_ids[0], 1,
        "positive post should rank higher than negative"
    );
}

#[test]
fn test_reddit_recsys_limit() {
    let users = make_users(2);
    let posts = make_posts(20);
    let recs = reddit::recommend(&users, &posts, 5);

    for user in &users {
        let user_recs: Vec<_> = recs.iter().filter(|(uid, _)| *uid == user.user_id).collect();
        assert_eq!(
            user_recs.len(),
            5,
            "user {} should get 5 recs, got {}",
            user.user_id,
            user_recs.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Twitter cosine-similarity
// ---------------------------------------------------------------------------

#[test]
fn test_twitter_recsys_personalized() {
    let users = vec![UserRow {
        user_id: 1,
        bio: "rust programming systems".into(),
    }];

    let posts = vec![
        PostRow {
            post_id: 10,
            user_id: 99,
            content: "rust programming language systems".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 0,
            num_dislikes: 0,
        },
        PostRow {
            post_id: 20,
            user_id: 99,
            content: "cooking recipe pasta dinner kitchen".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 0,
            num_dislikes: 0,
        },
    ];

    let traces: Vec<TraceRow> = vec![];
    let recs = twitter::recommend(&users, &posts, &traces, 2);
    let post_ids: Vec<i64> = recs.iter().map(|(_, pid)| *pid).collect();

    assert_eq!(
        post_ids[0], 10,
        "rust programming post should be recommended first to a rust programmer"
    );
}

#[test]
fn test_twitter_recsys_excludes_own_posts() {
    let users = vec![UserRow {
        user_id: 1,
        bio: "rust programming".into(),
    }];

    let posts = vec![
        PostRow {
            post_id: 10,
            user_id: 1, // user's own post
            content: "rust programming language".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 0,
            num_dislikes: 0,
        },
        PostRow {
            post_id: 20,
            user_id: 99,
            content: "cooking recipe".into(),
            created_at: "2026-04-01 12:00:00".into(),
            num_likes: 0,
            num_dislikes: 0,
        },
    ];

    let traces: Vec<TraceRow> = vec![];
    let recs = twitter::recommend(&users, &posts, &traces, 10);
    let rec_post_ids: HashSet<i64> = recs.iter().map(|(_, pid)| *pid).collect();

    assert!(
        !rec_post_ids.contains(&10),
        "user's own post (id=10) should not be recommended to them"
    );
    assert!(
        rec_post_ids.contains(&20),
        "other user's post (id=20) should be recommended"
    );
}

// ---------------------------------------------------------------------------
// TWHiN multi-signal
// ---------------------------------------------------------------------------

#[test]
fn test_twhin_recsys_with_likes() {
    let users = vec![UserRow {
        user_id: 1,
        bio: "technology enthusiast".into(),
    }];

    let ai_post = PostRow {
        post_id: 10,
        user_id: 99,
        content: "artificial intelligence deep learning neural networks".into(),
        created_at: "2026-04-01 12:00:00".into(),
        num_likes: 0,
        num_dislikes: 0,
    };

    let ml_post = PostRow {
        post_id: 20,
        user_id: 99,
        content: "machine learning neural networks deep learning AI".into(),
        created_at: "2026-04-01 12:00:00".into(),
        num_likes: 0,
        num_dislikes: 0,
    };

    let cooking_post = PostRow {
        post_id: 30,
        user_id: 99,
        content: "cooking recipe pasta dinner kitchen".into(),
        created_at: "2026-04-01 12:00:00".into(),
        num_likes: 0,
        num_dislikes: 0,
    };

    let posts = vec![ai_post, ml_post, cooking_post];

    // User liked the AI post
    let traces = vec![TraceRow {
        user_id: 1,
        action: "like_post".into(),
        info: "post_id: 10".into(),
    }];

    let recs = twhin::recommend(&users, &posts, &traces, 3);
    let post_ids: Vec<i64> = recs.iter().map(|(_, pid)| *pid).collect();

    // ML post (similar to liked AI post) should rank above cooking post
    let ml_rank = post_ids.iter().position(|&id| id == 20).unwrap();
    let cooking_rank = post_ids.iter().position(|&id| id == 30).unwrap();

    assert!(
        ml_rank < cooking_rank,
        "ML post (rank {}) should rank higher than cooking post (rank {}) due to like similarity with AI post",
        ml_rank,
        cooking_rank
    );
}

#[test]
fn test_twhin_empty_inputs() {
    let empty_users: Vec<UserRow> = vec![];
    let empty_posts: Vec<PostRow> = vec![];
    let traces: Vec<TraceRow> = vec![];

    let recs = twhin::recommend(&empty_users, &empty_posts, &traces, 10);
    assert!(recs.is_empty(), "empty inputs should produce empty results");
}
