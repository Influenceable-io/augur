/// Platform configuration options matching OASIS.
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Number of recommended posts returned per refresh.
    pub refresh_rec_post_count: usize,
    /// Maximum posts in the recommendation buffer per user.
    pub max_rec_post_len: usize,
    /// Show combined score instead of separate like/dislike counts.
    pub show_score: bool,
    /// Allow users to rate their own content.
    pub allow_self_rating: bool,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            refresh_rec_post_count: 10,
            max_rec_post_len: 1000,
            show_score: false,
            allow_self_rating: true,
        }
    }
}
