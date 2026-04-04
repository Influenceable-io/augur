/// Platform configuration options matching OASIS.
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Number of recommended posts returned per refresh.
    pub refresh_rec_post_count: usize,
    /// Number of posts from followed users returned per refresh.
    pub following_post_count: usize,
    /// Maximum posts in the recommendation buffer per user.
    pub max_rec_post_len: usize,
    /// Show combined score instead of separate like/dislike counts.
    pub show_score: bool,
    /// Allow users to rate their own content.
    pub allow_self_rating: bool,
    /// Number of top posts returned by the trend endpoint.
    pub trend_top_k: usize,
    /// Number of days to look back for trending posts.
    pub trend_num_days: usize,
    /// Number of reports before a post is auto-deleted.
    pub report_threshold: usize,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            refresh_rec_post_count: 10,
            following_post_count: 10,
            max_rec_post_len: 1000,
            show_score: false,
            allow_self_rating: true,
            trend_top_k: 10,
            trend_num_days: 3,
            report_threshold: 5,
        }
    }
}
