//! Shared HTTP helpers with realistic browser headers.

use ureq::Agent;

/// Real Chrome user agent string to avoid bot detection.
pub const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Build a ureq Agent with realistic browser defaults and timeouts.
///
/// Configures a 30-second connect timeout and 5-minute read timeout to
/// prevent stalled TCP connections from hanging forever.
#[must_use]
pub fn make_agent() -> Agent {
    use std::time::Duration;
    Agent::config_builder()
        .tls_config(tls_config())
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(300)))
        .build()
        .new_agent()
}

/// TLS in pure Rust, no C compiler: rustls with webpki's root certificates
/// and graviola's (`tls-graviola`) or RustCrypto's (`tls-rustcrypto`)
/// cryptography.
#[must_use]
pub fn tls_config() -> ureq::tls::TlsConfig {
    // graviola's cryptography where it runs and is enabled, else RustCrypto's
    // (always on wasm, where graviola has no backend)
    #[cfg(all(
        feature = "tls-graviola",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    let crypto = rustls_graviola::default_provider();
    #[cfg(all(
        any(feature = "tls-rustcrypto", target_family = "wasm"),
        not(all(
            feature = "tls-graviola",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))
    ))]
    let crypto = rustls_rustcrypto::provider();
    ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::Rustls)
        .root_certs(ureq::tls::RootCerts::WebPki)
        .unversioned_rustls_crypto_provider(std::sync::Arc::new(crypto))
        .build()
}

#[cfg(not(any(
    feature = "tls-rustcrypto",
    target_family = "wasm",
    all(
        feature = "tls-graviola",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
compile_error!(
    "https needs cryptography: enable `tls-graviola` (x86-64, aarch64) or `tls-rustcrypto` (any target; wasm always uses it)"
);

/// Common headers applied to every request.
#[must_use]
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

/// A failed request: ureq's error, and how long the server asked to be left
/// alone (`Retry-After`, on a 429 or 503) if it said.
pub(crate) struct Failed {
    pub(crate) error: ureq::Error,
    pub(crate) retry_after: Option<std::time::Duration>,
}

impl From<ureq::Error> for Failed {
    fn from(error: ureq::Error) -> Self {
        Self {
            error,
            retry_after: None,
        }
    }
}

impl Failed {
    /// An HTTP error status, with the response's `Retry-After` header value.
    pub(crate) fn status(status: u16, retry_after: Option<&str>) -> Self {
        Self {
            error: ureq::Error::StatusCode(status),
            retry_after: retry_after.and_then(crate::ratelimit::retry_after),
        }
    }
}

/// One GET: ureq's sockets, or on browser wasm (no sockets) the browser's
/// `XMLHttpRequest` (see [`crate::web`]'s notes)
fn send_get(agent: &Agent, url: &str) -> Result<ureq::http::Response<ureq::Body>, Failed> {
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    {
        let _ = agent;
        crate::web::send("GET", url, "*/*", None)
    }
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    {
        // statuses as responses, not errors: an error would drop the headers,
        // `Retry-After` among them
        let resp = get(agent, url)
            .config()
            .http_status_as_error(false)
            .build()
            .call()?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok());
            return Err(Failed::status(status, retry_after));
        }
        Ok(resp)
    }
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
    let mut last_err = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            crate::ratelimit::sleep(backoff(attempt));
        }

        // Rate limit
        if let Some(rl) = limiter {
            rl.acquire();
        }

        match send_get(agent, url) {
            Ok(resp) => {
                if let Some(rl) = limiter {
                    rl.clear_cooldown();
                }
                return Ok(resp);
            }
            Err(failed) => {
                if should_retry(&failed, limiter) && attempt < max_retries {
                    last_err = Some(failed.error);
                    continue;
                }
                return Err(crate::Error::Network(failed.error.to_string()));
            }
        }
    }
    Err(crate::Error::Network(
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "max retries exceeded".into()),
    ))
}

/// The wait before retry `attempt` (from 1): exponential, base, 2*base,
/// 4*base, ..., the base from `BIDS_RETRY_BASE_MS` (default 1000 ms).
pub(crate) fn backoff(attempt: u32) -> std::time::Duration {
    let base_ms: u64 = std::env::var("BIDS_RETRY_BASE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    std::time::Duration::from_millis(base_ms * (1 << (attempt - 1)))
}

/// Whether a failed request is worth retrying: 429, 500, 502, 503, 504 and
/// network errors are. A 429 also starts a cooldown, for the server's
/// `Retry-After` if it sent one, else the limiter's default (30 s without a
/// limiter, waited here); a 5xx with `Retry-After` does too.
pub(crate) fn should_retry(
    failed: &Failed,
    limiter: Option<&crate::ratelimit::RateLimiter>,
) -> bool {
    let cool_down = |wait: Option<std::time::Duration>| match limiter {
        Some(rl) => rl.on_rate_limited(wait),
        None => crate::ratelimit::sleep(wait.unwrap_or(std::time::Duration::from_secs(30))),
    };
    match failed.error {
        ureq::Error::StatusCode(429) => {
            cool_down(failed.retry_after);
            true
        }
        ureq::Error::StatusCode(500 | 502 | 503 | 504) => {
            if failed.retry_after.is_some() {
                cool_down(failed.retry_after);
            }
            true
        }
        ureq::Error::StatusCode(_) => false,
        _ => true,
    }
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
    #[must_use]
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

    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    let mut resp = {
        let _ = agent;
        crate::web::send("POST", url, "application/json", Some(&json_str))
            .map_err(|f| crate::Error::Network(f.error.to_string()))?
    };
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
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

#[cfg(all(test, not(all(target_family = "wasm", target_os = "unknown"))))]
mod tests {
    use super::*;
    use crate::ratelimit::RateLimiter;
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};

    /// A local server answering its connections with `responses`, in order.
    fn serve(responses: &[&'static str]) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let responses = responses.to_vec();
        std::thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        url
    }

    const OK: &str = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";

    fn body(resp: ureq::http::Response<ureq::Body>) -> String {
        resp.into_body().read_to_string().unwrap()
    }

    #[test]
    fn a_429_waits_for_its_retry_after() {
        let url = serve(&[
            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            OK,
        ]);
        // a 60 s default cooldown: honoring the server's 1 s is what's quick
        let limiter = RateLimiter::new_with_cooldown(0.0, 60);
        let start = Instant::now();
        let resp = get_with_retry_limited(&make_agent(), &url, 3, Some(&limiter)).unwrap();
        assert_eq!(body(resp), "ok");
        let took = start.elapsed();
        assert!(
            took >= Duration::from_millis(900) && took < Duration::from_secs(10),
            "{took:?}"
        );
    }

    #[test]
    fn a_503_with_retry_after_waits_for_it() {
        let url = serve(&[
            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            OK,
        ]);
        let limiter = RateLimiter::new_with_cooldown(0.0, 60);
        let start = Instant::now();
        let resp = get_with_retry_limited(&make_agent(), &url, 3, Some(&limiter)).unwrap();
        assert_eq!(body(resp), "ok");
        // the 2 s asked for, not just the 1 s backoff
        let took = start.elapsed();
        assert!(
            took >= Duration::from_millis(1900) && took < Duration::from_secs(10),
            "{took:?}"
        );
    }

    #[test]
    fn a_429_without_retry_after_takes_the_default_cooldown() {
        let url = serve(&[
            "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            OK,
        ]);
        let limiter = RateLimiter::new_with_cooldown(0.0, 2);
        let start = Instant::now();
        let resp = get_with_retry_limited(&make_agent(), &url, 3, Some(&limiter)).unwrap();
        assert_eq!(body(resp), "ok");
        assert!(
            start.elapsed() >= Duration::from_millis(1900),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn a_404_is_not_retried() {
        // one response only: a retry would find nobody listening
        let url =
            serve(&["HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"]);
        let start = Instant::now();
        let e = get_with_retry_limited(&make_agent(), &url, 3, None).unwrap_err();
        assert!(e.to_string().contains("404"), "{e}");
        assert!(
            start.elapsed() < Duration::from_millis(900),
            "{:?}",
            start.elapsed()
        );
    }
}
