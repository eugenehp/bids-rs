//! Rate limiter for HTTP requests to avoid 429 responses.
//!
//! Implements a token-bucket rate limiter shared across all download threads.
//! Also parses `Retry-After` headers from 429 responses for server-directed backoff.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
// browser wasm has no clock in std (`Instant::now` panics); web-time's is
// `performance.now()` there and std's everywhere else
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::time::Instant;
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
use web_time::Instant;

/// Block for `d`. Browser wasm can't sleep with std (it panics there): see
/// `web::sleep` for what it does instead.
pub(crate) fn sleep(d: Duration) {
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    std::thread::sleep(d);
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    crate::web::sleep(d);
}

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
    #[must_use]
    pub fn new(max_rps: f64) -> Self {
        Self::new_with_cooldown(max_rps, 30)
    }

    /// Create with a custom default 429 cooldown (seconds).
    #[must_use]
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
    #[must_use]
    pub fn unlimited() -> Self {
        Self::new(0.0)
    }

    /// Wait until we're allowed to make a request.
    ///
    /// Returns immediately if within rate limit, otherwise sleeps.
    pub fn acquire(&self) {
        while let Err(wait) = self.try_acquire() {
            sleep(wait);
        }
    }

    /// [`acquire`](Self::acquire) without blocking the thread: on browser
    /// wasm, other requests in flight carry on while this one waits.
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    pub(crate) async fn acquire_async(&self) {
        while let Err(wait) = self.try_acquire() {
            crate::web::sleep_async(wait).await;
        }
    }

    /// Claim a request slot now, or say how long until one may be free.
    fn try_acquire(&self) -> Result<(), Duration> {
        // Check global cooldown (from 429 response)
        if let Ok(guard) = self.cooldown_until.lock() {
            if let Some(until) = *guard {
                let now = Instant::now();
                if now < until {
                    return Err(until - now);
                }
            }
        }

        if self.min_interval_us == 0 {
            return Ok(());
        }

        loop {
            let now_us = self.epoch.elapsed().as_micros() as u64;
            let last = self.last_request_us.load(Ordering::Relaxed);
            let earliest = last + self.min_interval_us;

            if now_us >= earliest {
                // Try to claim this slot
                if self
                    .last_request_us
                    .compare_exchange(last, now_us, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    return Ok(());
                }
                // Another thread beat us — retry
                continue;
            }

            // Need to wait
            return Err(Duration::from_micros(earliest - now_us));
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
                _ => {
                    *guard = Some(until);
                }
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
/// Supports both seconds (`Retry-After: 120`) and an HTTP-date
/// (`Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`, IMF-fixdate, the form
/// senders must use), as the time left until then: zero once it's past.
#[must_use]
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    // Try as integer seconds first
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // else an HTTP-date, which is how long from now
    let at = parse_http_date(value)?;
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    use std::time::{SystemTime, UNIX_EPOCH};
    // (std's clock panics on browser wasm)
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    use web_time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(Duration::from_secs(at.saturating_sub(now)))
}

/// The longest `Retry-After` honored: a server asking for longer (or a
/// mistaken date) still gets retried this soon, rather than hanging a
/// download for hours.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(10 * 60);

/// A response's `Retry-After` header value as a wait, at most
/// [`MAX_RETRY_AFTER`].
pub(crate) fn retry_after(value: &str) -> Option<Duration> {
    parse_retry_after(value).map(|d| d.min(MAX_RETRY_AFTER))
}

/// An HTTP-date in its required form, IMF-fixdate
/// (`Sun, 06 Nov 1994 08:49:37 GMT`), as seconds since the Unix epoch.
fn parse_http_date(value: &str) -> Option<u64> {
    let mut parts = value.split_ascii_whitespace();
    let _weekday = parts.next()?.strip_suffix(',')?;
    let day: u64 = parts.next()?.parse().ok()?;
    let month = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut hms = parts.next()?.split(':').map(str::parse::<u64>);
    let (h, m, sec) = (hms.next()?.ok()?, hms.next()?.ok()?, hms.next()?.ok()?);
    if parts.next()? != "GMT" || parts.next().is_some() || hms.next().is_some() {
        return None;
    }
    if !(1..=31).contains(&day) || h > 23 || m > 59 || sec > 60 || year < 1970 {
        return None;
    }
    // days since 1970-01-01 of a proleptic Gregorian date (Hinnant's
    // days_from_civil), the year starting in March so Feb 29 comes last
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(u64::try_from(days).ok()? * 86_400 + h * 3600 + m * 60 + sec)
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
    #[must_use]
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
    #[must_use]
    pub fn s3_download_rps(mut self, rps: f64) -> Self {
        self.s3_download_rps = rps;
        self
    }
    /// Builder: set S3 listing rate.
    #[must_use]
    pub fn s3_listing_rps(mut self, rps: f64) -> Self {
        self.s3_listing_rps = rps;
        self
    }
    /// Builder: set GraphQL rate.
    #[must_use]
    pub fn graphql_rps(mut self, rps: f64) -> Self {
        self.graphql_rps = rps;
        self
    }
    /// Builder: set max retries.
    #[must_use]
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }
    /// Builder: set retry base delay.
    #[must_use]
    pub fn retry_base_ms(mut self, ms: u64) -> Self {
        self.retry_base_ms = ms;
        self
    }
    /// Builder: set 429 cooldown.
    #[must_use]
    pub fn default_429_cooldown_secs(mut self, secs: u64) -> Self {
        self.default_429_cooldown_secs = secs;
        self
    }
    /// Builder: set download thread count.
    #[must_use]
    pub fn download_threads(mut self, n: usize) -> Self {
        self.download_threads = n;
        self
    }

    /// Create the S3 download rate limiter from this config.
    #[must_use]
    pub fn s3_download_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.s3_download_rps, self.default_429_cooldown_secs)
    }
    /// Create the S3 listing rate limiter from this config.
    #[must_use]
    pub fn s3_listing_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.s3_listing_rps, self.default_429_cooldown_secs)
    }
    /// Create the GraphQL rate limiter from this config.
    #[must_use]
    pub fn graphql_limiter(&self) -> RateLimiter {
        RateLimiter::new_with_cooldown(self.graphql_rps, self.default_429_cooldown_secs)
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Default rate limits (reads from env vars, falls back to safe defaults).
pub mod defaults {
    use super::*;

    /// Default config (from env vars or hardcoded defaults).
    #[must_use]
    pub fn config() -> RateLimitConfig {
        RateLimitConfig::from_env()
    }

    /// Rate limiter for S3 downloads.
    #[must_use]
    pub fn s3_limiter() -> RateLimiter {
        config().s3_download_limiter()
    }

    /// Rate limiter for S3 listing.
    #[must_use]
    pub fn s3_listing_limiter() -> RateLimiter {
        config().s3_listing_limiter()
    }

    /// Rate limiter for GraphQL API.
    #[must_use]
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
        assert!(
            elapsed >= Duration::from_millis(30),
            "5 requests took {:?}, expected >= 30ms",
            elapsed
        );
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RateLimiter::unlimited();
        let start = Instant::now();
        for _ in 0..100 {
            limiter.acquire();
        }
        // Should not wait at all: allow for a busy machine, not for a limit
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn test_cooldown() {
        let limiter = RateLimiter::new(1000.0);

        // Set a 100ms cooldown
        limiter.on_rate_limited(Some(Duration::from_millis(100)));

        let start = Instant::now();
        limiter.acquire();
        let elapsed = start.elapsed();

        assert!(
            elapsed >= Duration::from_millis(80),
            "Cooldown acquire took {:?}, expected >= 80ms",
            elapsed
        );

        // Clear and verify
        limiter.clear_cooldown();
        // the 100ms cooldown is gone: well under it, with room for a busy machine
        let start2 = Instant::now();
        limiter.acquire();
        assert!(start2.elapsed() < Duration::from_millis(75));
    }

    #[test]
    fn test_parse_retry_after() {
        assert_eq!(parse_retry_after("120"), Some(Duration::from_secs(120)));
        assert_eq!(parse_retry_after("  30  "), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("not-a-number"), None);
        // an HTTP-date long gone: no wait
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(Duration::ZERO)
        );
        // one in the future: about that long, and capped when it's far
        let soon = std::time::SystemTime::now() + Duration::from_secs(120);
        let wait = parse_retry_after(&http_date(soon)).unwrap();
        assert!((118..=120).contains(&wait.as_secs()), "{wait:?}");
        assert_eq!(
            retry_after("Fri, 01 Jan 2100 00:00:00 GMT"),
            Some(MAX_RETRY_AFTER)
        );
        assert_eq!(retry_after("90"), Some(Duration::from_secs(90)));
    }

    #[test]
    fn http_dates() {
        assert_eq!(parse_http_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        // RFC 9110's example, and a leap day
        assert_eq!(
            parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(
            parse_http_date("Thu, 29 Feb 2024 12:00:00 GMT"),
            Some(1_709_208_000)
        );
        for bad in [
            "",
            "06 Nov 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 08:49:37 UTC",
            "Sun, 06 Foo 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 08:49 GMT",
            "Sun, 06 Nov 1994 25:49:37 GMT",
            "Sun, 06 Nov 1994 08:49:37 GMT extra",
        ] {
            assert_eq!(parse_http_date(bad), None, "{bad:?}");
        }
    }

    /// `t` as an IMF-fixdate, for the tests (the inverse of the parser).
    fn http_date(t: std::time::SystemTime) -> String {
        let secs = t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
        // Hinnant's civil_from_days
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        let names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        format!(
            "Mon, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
            names[(month - 1) as usize],
            rem / 3600,
            rem / 60 % 60,
            rem % 60
        )
    }
}
