//! Rate limiter for HTTP requests to avoid 429 responses.
//!
//! Implements a token-bucket rate limiter shared across all download threads.
//! Also parses `Retry-After` headers from 429 responses for server-directed backoff.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A thread-safe token-bucket rate limiter.
///
/// Limits requests to `max_rps` requests per second across all threads.
/// When the bucket is empty, callers block until a token is available.
pub struct RateLimiter {
    /// Minimum interval between requests in microseconds.
    min_interval_us: u64,
    /// Timestamp of the last request (microseconds since process start).
    last_request_us: AtomicU64,
    /// Start time for converting Instant to u64.
    epoch: Instant,
    /// Lock for cooldown periods (429 backoff).
    cooldown_until: Mutex<Option<Instant>>,
    /// Default cooldown duration when 429 has no Retry-After header.
    default_cooldown: Duration,
}

impl RateLimiter {
    /// Create a rate limiter allowing `max_rps` requests per second.
    pub fn new(max_rps: f64) -> Self {
        Self::new_with_cooldown(max_rps, 30)
    }

    /// Create with a custom default 429 cooldown (seconds).
    pub fn new_with_cooldown(max_rps: f64, default_cooldown_secs: u64) -> Self {
        let min_interval_us = if max_rps > 0.0 {
            (1_000_000.0 / max_rps) as u64
        } else {
            0
        };
        Self {
            min_interval_us,
            last_request_us: AtomicU64::new(0),
            epoch: Instant::now(),
            cooldown_until: Mutex::new(None),
            default_cooldown: Duration::from_secs(default_cooldown_secs),
        }
    }

    /// Unlimited rate limiter (no throttling).
    pub fn unlimited() -> Self {
        Self::new(0.0)
    }

    /// Wait until we're allowed to make a request.
    ///
    /// Returns immediately if within rate limit, otherwise sleeps.
    pub fn acquire(&self) {
        // Check global cooldown (from 429 response)
        if let Ok(guard) = self.cooldown_until.lock() {
            if let Some(until) = *guard {
                let now = Instant::now();
                if now < until {
                    let wait = until - now;
                    drop(guard); // release lock before sleeping
                    std::thread::sleep(wait);
                }
            }
        }

        if self.min_interval_us == 0 { return; }

        loop {
            let now_us = self.epoch.elapsed().as_micros() as u64;
            let last = self.last_request_us.load(Ordering::Relaxed);
            let earliest = last + self.min_interval_us;

            if now_us >= earliest {
                // Try to claim this slot
                if self.last_request_us
                    .compare_exchange(last, now_us, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
                // Another thread beat us — retry
                continue;
            }

            // Need to wait
            let wait_us = earliest - now_us;
            std::thread::sleep(Duration::from_micros(wait_us));
        }
    }

    /// Notify the rate limiter that we got a 429 response.
    ///
    /// If `retry_after` is provided (from the `Retry-After` header), all
    /// threads will pause for that duration. Otherwise uses the configured
    /// default cooldown (env `BIDS_429_COOLDOWN_SECS`, default 30s).
    pub fn on_rate_limited(&self, retry_after: Option<Duration>) {
        let wait = retry_after.unwrap_or(self.default_cooldown);
        let until = Instant::now() + wait;
        if let Ok(mut guard) = self.cooldown_until.lock() {
            // Only extend the cooldown, never shorten it
            match *guard {
                Some(existing) if existing > until => {}
                _ => { *guard = Some(until); }
            }
        }
    }

    /// Clear any active cooldown (e.g., after successful request post-429).
    pub fn clear_cooldown(&self) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            *guard = None;
        }
    }
}

/// Parse a `Retry-After` header value into a Duration.
///
/// Supports both seconds (`Retry-After: 120`) and HTTP-date formats
/// (though we only handle the seconds format for simplicity).
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    // Try as integer seconds first
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // Could parse HTTP-date here, but seconds is most common for APIs
    None
}

/// Configuration for all rate limits, readable from environment variables or code.
///
/// # Environment variables
///
/// | Variable | Default | Description |
/// |---|---|---|
/// | `BIDS_S3_DOWNLOAD_RPS` | 50 | Max S3 download requests/second |
/// | `BIDS_S3_LISTING_RPS` | 10 | Max S3 listing requests/second |
/// | `BIDS_GRAPHQL_RPS` | 5 | Max GraphQL API requests/second |
/// | `BIDS_MAX_RETRIES` | 3 | Max retry attempts per request |
/// | `BIDS_RETRY_BASE_MS` | 1000 | Base delay for exponential backoff (ms) |
/// | `BIDS_429_COOLDOWN_SECS` | 30 | Default cooldown on 429 (when no Retry-After header) |
/// | `BIDS_DOWNLOAD_THREADS` | 8 | Number of parallel download threads |
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Max S3 download requests per second (0 = unlimited).
    pub s3_download_rps: f64,
    /// Max S3 listing requests per second.
    pub s3_listing_rps: f64,
    /// Max GraphQL API requests per second.
    pub graphql_rps: f64,
    /// Max retry attempts per request.
    pub max_retries: u32,
    /// Base delay for exponential backoff in milliseconds.
    pub retry_base_ms: u64,
    /// Default 429 cooldown in seconds (when no Retry-After header).
    pub default_429_cooldown_secs: u64,
    /// Number of parallel download threads.
    pub download_threads: usize,
}

impl RateLimitConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        Self {
            s3_download_rps: env_f64("BIDS_S3_DOWNLOAD_RPS", 50.0),
            s3_listing_rps: env_f64("BIDS_S3_LISTING_RPS", 10.0),
            graphql_rps: env_f64("BIDS_GRAPHQL_RPS", 5.0),
            max_retries: env_u32("BIDS_MAX_RETRIES", 3),
            retry_base_ms: env_u64("BIDS_RETRY_BASE_MS", 1000),
            default_429_cooldown_secs: env_u64("BIDS_429_COOLDOWN_SECS", 30),
            download_threads: env_usize("BIDS_DOWNLOAD_THREADS", 8),
        }
    }

    /// Builder: set S3 download rate.
    #[must_use] pub fn s3_download_rps(mut self, rps: f64) -> Self { self.s3_download_rps = rps; self }
    /// Builder: set S3 listing rate.
    #[must_use] pub fn s3_listing_rps(mut self, rps: f64) -> Self { self.s3_listing_rps = rps; self }
    /// Builder: set GraphQL rate.
    #[must_use] pub fn graphql_rps(mut self, rps: f64) -> Self { self.graphql_rps = rps; self }
    /// Builder: set max retries.
    #[must_use] pub fn max_retries(mut self, n: u32) -> Self { self.max_retries = n; self }
    /// Builder: set retry base delay.
    #[must_use] pub fn retry_base_ms(mut self, ms: u64) -> Self { self.retry_base_ms = ms; self }
    /// Builder: set 429 cooldown.
    #[must_use] pub fn default_429_cooldown_secs(mut self, secs: u64) -> Self { self.default_429_cooldown_secs = secs; self }
    /// Builder: set download thread count.
    #[must_use] pub fn download_threads(mut self, n: usize) -> Self { self.download_threads = n; self }

    /// Create the S3 download rate limiter from this config.
    pub fn s3_download_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.s3_download_rps, self.default_429_cooldown_secs)
    }
    /// Create the S3 listing rate limiter from this config.
    pub fn s3_listing_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.s3_listing_rps, self.default_429_cooldown_secs)
    }
    /// Create the GraphQL rate limiter from this config.
    pub fn graphql_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.graphql_rps, self.default_429_cooldown_secs)
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self { Self::from_env() }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Default rate limits (reads from env vars, falls back to safe defaults).
pub mod defaults {
    use super::*;

    /// Default config (from env vars or hardcoded defaults).
    pub fn config() -> RateLimitConfig {
        RateLimitConfig::from_env()
    }

    /// Rate limiter for S3 downloads.
    pub fn s3_limiter() -> RateLimiter {
        config().s3_download_limiter()
    }

    /// Rate limiter for S3 listing.
    pub fn s3_listing_limiter() -> RateLimiter {
        config().s3_listing_limiter()
    }

    /// Rate limiter for GraphQL API.
    pub fn graphql_limiter() -> RateLimiter {
        config().graphql_limiter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(100.0); // 100 req/s = 10ms between requests

        let start = Instant::now();
        for _ in 0..5 {
            limiter.acquire();
        }
        let elapsed = start.elapsed();

        // 5 requests at 100 req/s should take at least ~40ms (4 intervals)
        assert!(elapsed >= Duration::from_millis(30),
            "5 requests took {:?}, expected >= 30ms", elapsed);
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RateLimiter::unlimited();
        let start = Instant::now();
        for _ in 0..100 {
            limiter.acquire();
        }
        // Should be near-instant
        assert!(start.elapsed() < Duration::from_millis(10));
    }

    #[test]
    fn test_cooldown() {
        let limiter = RateLimiter::new(1000.0);

        // Set a 100ms cooldown
        limiter.on_rate_limited(Some(Duration::from_millis(100)));

        let start = Instant::now();
        limiter.acquire();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(80),
            "Cooldown acquire took {:?}, expected >= 80ms", elapsed);

        // Clear and verify
        limiter.clear_cooldown();
        let start2 = Instant::now();
        limiter.acquire();
        assert!(start2.elapsed() < Duration::from_millis(20));
    }

    #[test]
    fn test_parse_retry_after() {
        assert_eq!(parse_retry_after("120"), Some(Duration::from_secs(120)));
        assert_eq!(parse_retry_after("  30  "), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("not-a-number"), None);
    }
}
