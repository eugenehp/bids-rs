//! Shared HTTP helpers with realistic browser headers.

use ureq::Agent;

/// Real Chrome user agent string to avoid bot detection.
pub const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Build a ureq Agent with realistic browser defaults and timeouts.
///
/// Configures a 30-second connect timeout and 5-minute read timeout to
/// prevent stalled TCP connections from hanging forever.
pub fn make_agent() -> Agent {
    use std::time::Duration;
    Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(300)))
        .build()
        .new_agent()
}

/// Common headers applied to every request.
pub fn get(agent: &Agent, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    agent
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "*/*")
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("Accept-Encoding", "identity")
        .header("Connection", "keep-alive")
        .header("DNT", "1")
}

/// GET with rate limiting and retry on transient failures.
///
/// - Acquires a token from the rate limiter before each request
/// - On 429: parses Retry-After header and triggers global cooldown
/// - On 5xx / network errors: exponential backoff (1s, 2s, 4s, ...)
pub fn get_with_retry(
    agent: &Agent,
    url: &str,
    max_retries: u32,
) -> crate::Result<ureq::http::Response<ureq::Body>> {
    get_with_retry_limited(agent, url, max_retries, None)
}

/// GET with explicit rate limiter.
///
/// Retry base delay is controlled by `BIDS_RETRY_BASE_MS` env var (default 1000ms).
pub fn get_with_retry_limited(
    agent: &Agent,
    url: &str,
    max_retries: u32,
    limiter: Option<&crate::ratelimit::RateLimiter>,
) -> crate::Result<ureq::http::Response<ureq::Body>> {
    let base_ms: u64 = std::env::var("BIDS_RETRY_BASE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let mut last_err = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            // Exponential backoff: base, 2*base, 4*base, ...
            std::thread::sleep(std::time::Duration::from_millis(
                base_ms * (1 << (attempt - 1)),
            ));
        }

        // Rate limit
        if let Some(rl) = limiter {
            rl.acquire();
        }

        match get(agent, url).call() {
            Ok(resp) => {
                if let Some(rl) = limiter {
                    rl.clear_cooldown();
                }
                return Ok(resp);
            }
            Err(e) => {
                let should_retry = match &e {
                    ureq::Error::StatusCode(429) => {
                        // Rate limited — parse Retry-After and trigger cooldown
                        if let Some(rl) = limiter {
                            // ureq::Error doesn't give us headers on error,
                            // so use default 30s cooldown
                            rl.on_rate_limited(None);
                        } else {
                            // No rate limiter — just sleep 30s
                            std::thread::sleep(std::time::Duration::from_secs(30));
                        }
                        true
                    }
                    ureq::Error::StatusCode(code) => matches!(code, 500 | 502 | 503 | 504),
                    _ => true,
                };
                if should_retry && attempt < max_retries {
                    last_err = Some(e);
                    continue;
                }
                return Err(crate::Error::Network(e.to_string()));
            }
        }
    }
    Err(crate::Error::Network(
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "max retries exceeded".into()),
    ))
}

/// Trait for download progress callbacks.
pub trait ProgressCallback: Send {
    fn on_file_start(&mut self, path: &str, size: u64);
    fn on_file_done(&mut self, path: &str);
    fn on_error(&mut self, path: &str, err: &str);
}

/// Simple stderr progress reporter.
pub struct StderrProgress {
    total: usize,
    done: usize,
}

impl StderrProgress {
    pub fn new(total: usize) -> Self {
        Self { total, done: 0 }
    }
}

impl ProgressCallback for StderrProgress {
    fn on_file_start(&mut self, path: &str, _size: u64) {
        self.done += 1;
        eprint!("\r  [{}/{}] {}", self.done, self.total, path);
    }
    fn on_file_done(&mut self, _path: &str) {}
    fn on_error(&mut self, path: &str, err: &str) {
        eprintln!("\n  ERROR {path}: {err}");
    }
}

/// POST with JSON body, browser-like headers, and optional rate limiting.
pub fn post_json(
    agent: &Agent,
    url: &str,
    body: &serde_json::Value,
) -> crate::Result<serde_json::Value> {
    post_json_limited(agent, url, body, None)
}

/// POST with explicit rate limiter.
pub fn post_json_limited(
    agent: &Agent,
    url: &str,
    body: &serde_json::Value,
    limiter: Option<&crate::ratelimit::RateLimiter>,
) -> crate::Result<serde_json::Value> {
    if let Some(rl) = limiter {
        rl.acquire();
    }
    let json_str = serde_json::to_string(body).map_err(|e| crate::Error::Network(e.to_string()))?;

    let mut resp = agent
        .post(url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/json")
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("Content-Type", "application/json")
        .header("Origin", "https://openneuro.org")
        .header("Referer", "https://openneuro.org/")
        .header("DNT", "1")
        .header("Sec-Fetch-Dest", "empty")
        .header("Sec-Fetch-Mode", "cors")
        .header("Sec-Fetch-Site", "same-origin")
        .send(json_str.as_bytes())
        .map_err(|e| crate::Error::Network(e.to_string()))?;

    let json: serde_json::Value = resp
        .body_mut()
        .read_json()
        .map_err(|e| crate::Error::Network(format!("JSON parse: {e}")))?;
    Ok(json)
}
