//! https on browser wasm (`wasm32-unknown-unknown`), where there are no
//! sockets: the browser's own synchronous `XMLHttpRequest`, so the blocking
//! API stays as it is. OpenNeuro's S3 bucket and GraphQL endpoint both allow
//! cross-origin requests.
//!
//! A synchronous request blocks the thread it runs on: call from a Web Worker,
//! not the page's main thread, which it would freeze. In a worker, requests
//! also time out (the page's synchronous ones can't), and waits sleep
//! instead of spinning when the page is cross-origin isolated.
//!
//! [`OpenNeuro::download_to_opfs`](crate::OpenNeuro::download_to_opfs) is
//! the async side: `fetch`, several requests in flight at once, async
//! waits, so one request's backoff doesn't hold up the rest, and bodies
//! streamed to OPFS, so memory doesn't grow with file size.

use std::time::Duration;

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::http::Failed;
use web_sys::{XmlHttpRequest, XmlHttpRequestResponseType};

/// A request's time limit: native's connect (30 s) and body (5 min) limits
/// together, as the browser has only the one.
const TIMEOUT: Duration = Duration::from_secs(30 + 300);

/// Whether this runs on a page (a `Window`) rather than in a worker.
pub(crate) fn in_window() -> bool {
    js_sys::global().dyn_into::<web_sys::Window>().is_ok()
}

/// Block for `d`, which std can't do here (its sleep panics). A worker on a
/// cross-origin-isolated page (COOP `same-origin` + COEP `require-corp`) has
/// `SharedArrayBuffer`, so `Atomics.wait` sleeps the thread; anywhere else
/// (a page's main thread may not wait at all) it spins on the clock.
pub(crate) fn sleep(d: Duration) {
    thread_local! {
        static WAIT_CELL: Option<js_sys::Int32Array> = {
            let global = js_sys::global();
            let isolated = js_sys::Reflect::get(&global, &"crossOriginIsolated".into())
                .is_ok_and(|v| v.is_truthy());
            (isolated && !in_window())
                .then(|| js_sys::Int32Array::new(&js_sys::SharedArrayBuffer::new(4)))
        };
    }
    let waited = WAIT_CELL.with(|cell| {
        cell.as_ref().is_some_and(|cell| {
            // nobody notifies the cell, so this always runs to its timeout
            js_sys::Atomics::wait_with_timeout(cell, 0, 0, d.as_secs_f64() * 1000.0).is_ok()
        })
    });
    if !waited {
        let until = web_time::Instant::now() + d;
        while web_time::Instant::now() < until {
            std::hint::spin_loop();
        }
    }
}

/// Wait `d` without blocking: a `setTimeout`, so the event loop (and every
/// other request in flight) runs meanwhile.
pub(crate) async fn sleep_async(d: Duration) {
    let ms = d.as_millis().min(i32::MAX as u128) as i32;
    let timer = js_sys::Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into())
            .and_then(JsCast::dyn_into::<js_sys::Function>);
        match set_timeout {
            Ok(set_timeout) => {
                let _ = set_timeout.call2(&global, &resolve, &ms.into());
            }
            // no timers at all: don't wait
            Err(_) => {
                let _ = resolve.call0(&JsValue::UNDEFINED);
            }
        }
    });
    let _ = JsFuture::from(timer).await;
}

/// One request, failing with ureq's errors so that `http`'s retry logic
/// (429, 5xx, network) treats both transports alike.
///
/// Only CORS-safe headers are set: browsers refuse `User-Agent`, `Origin`,
/// `Referer`, `DNT`, `Sec-*`, `Connection` and `Accept-Encoding` from scripts,
/// and send their own.
pub(crate) fn send(
    method: &str,
    url: &str,
    accept: &str,
    json: Option<&str>,
) -> Result<ureq::http::Response<ureq::Body>, Failed> {
    let xhr = XmlHttpRequest::new().map_err(js_error)?;
    xhr.open_with_async(method, url, false).map_err(js_error)?;
    // binary bodies: a worker may ask a synchronous request for an
    // ArrayBuffer; a window may not, so there each byte comes back as one
    // char of the `x-user-defined` charset (byte b as U+F700 + b, b < 0x80
    // as itself), whose low byte is b
    let in_window = in_window();
    if in_window {
        xhr.override_mime_type("text/plain; charset=x-user-defined")
            .map_err(js_error)?;
    } else {
        xhr.set_response_type(XmlHttpRequestResponseType::Arraybuffer);
        // a page's synchronous requests can't time out; a worker's can
        xhr.set_timeout(TIMEOUT.as_millis() as u32);
    }
    xhr.set_request_header("Accept", accept).map_err(js_error)?;
    match json {
        Some(body) => {
            xhr.set_request_header("Content-Type", "application/json")
                .map_err(js_error)?;
            xhr.send_with_opt_str(Some(body)).map_err(js_error)?;
        }
        None => xhr.send().map_err(js_error)?,
    }
    // (a timeout throws from `send`: `js_error` makes it `TimedOut`)

    let status = xhr.status().map_err(js_error)?;
    // 0: no response at all, a network error or a cross-origin refusal
    if status == 0 {
        return Err(ureq::Error::Io(std::io::Error::other(format!(
            "{method} {url}: no response (network error, or the server refused a cross-origin request)"
        )))
        .into());
    }
    if status >= 400 {
        // readable only if the server exposes it to scripts (CORS:
        // `Access-Control-Expose-Headers: Retry-After`)
        let retry_after = xhr.get_response_header("Retry-After").ok().flatten();
        return Err(Failed::status(status, retry_after.as_deref()));
    }

    let data: Vec<u8> = if in_window {
        let text = xhr.response_text().map_err(js_error)?.unwrap_or_default();
        text.chars().map(|c| c as u32 as u8).collect()
    } else {
        js_sys::Uint8Array::new(&xhr.response().map_err(js_error)?).to_vec()
    };

    let mut body = ureq::Body::builder();
    if let Ok(Some(content_type)) = xhr.get_response_header("Content-Type") {
        let mut parts = content_type.split(';').map(str::trim);
        if let Some(mime) = parts.next().filter(|m| !m.is_empty()) {
            body = body.mime_type(mime);
        }
        if let Some(charset) = parts.find_map(|p| p.strip_prefix("charset=")) {
            body = body.charset(charset.trim_matches('"'));
        }
    }
    Ok(ureq::http::Response::builder()
        .status(status)
        .body(body.data(data))
        .map_err(ureq::Error::Http)?)
}

/// GET `url` with `fetch` into the OPFS file `path` under `dir`, streaming:
/// each chunk goes from the response to the file as it arrives, never all of
/// it in memory (nor through wasm's), so a file's size is the disk's
/// business only. Retries as [`crate::http`] does (same policy, same
/// backoff), waiting asynchronously, each try from the start; a body shorter
/// or longer than `expected` is a failed try too. Returns the bytes written.
pub(crate) async fn fetch_to_opfs(
    url: &str,
    max_retries: u32,
    limiter: &crate::ratelimit::RateLimiter,
    dir: &web_sys::FileSystemDirectoryHandle,
    path: &str,
    expected: Option<u64>,
) -> crate::Result<u64> {
    let mut attempt = 0;
    loop {
        if attempt > 0 {
            sleep_async(crate::http::backoff(attempt)).await;
        }
        limiter.acquire_async().await;
        match stream_once(url, dir, path, expected).await {
            Ok(written) => {
                limiter.clear_cooldown();
                return Ok(written);
            }
            Err(Try::Http(failed))
                if attempt < max_retries && crate::http::should_retry(&failed, Some(limiter)) =>
            {
                attempt += 1;
            }
            Err(Try::Http(failed)) => return Err(crate::Error::Network(failed.error.to_string())),
            // the disk (quota, a locked file) won't be better on a retry
            Err(Try::Disk(e)) => return Err(e),
        }
    }
}

/// How a try at [`fetch_to_opfs`] failed: the request (perhaps worth
/// retrying) or the disk.
enum Try {
    Http(Failed),
    Disk(crate::Error),
}

/// One `fetch` streamed to the file, with [`TIMEOUT`]. On failure the file
/// is removed, so a resumed download doesn't take it for finished.
async fn stream_once(
    url: &str,
    dir: &web_sys::FileSystemDirectoryHandle,
    path: &str,
    expected: Option<u64>,
) -> Result<u64, Try> {
    let response = fetch_response(url).await.map_err(Try::Http)?;
    let mut file = crate::opfs::Writer::create(dir, path)
        .await
        .map_err(Try::Disk)?;
    let pumped = pump(&response, &mut file).await;
    let checked = pumped.and_then(|()| match expected {
        Some(size) if file.written() != size => Err(Try::Http(
            ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("{url}: got {} bytes, expected {size}", file.written()),
            ))
            .into(),
        )),
        _ => Ok(()),
    });
    match checked {
        Ok(()) => file.finish().await.map_err(Try::Disk),
        Err(e) => {
            file.abandon().await;
            Err(e)
        }
    }
}

/// The response body's chunks into `file`, as they arrive.
async fn pump(response: &web_sys::Response, file: &mut crate::opfs::Writer) -> Result<(), Try> {
    // no body at all: an empty file
    let Some(body) = response.body() else {
        return Ok(());
    };
    let reader: web_sys::ReadableStreamDefaultReader = body.get_reader().unchecked_into();
    loop {
        // rejects when the connection breaks or the timeout aborts it
        let next: js_sys::IteratorNext = JsFuture::from(reader.read())
            .await
            .map_err(|e| Try::Http(js_error(e).into()))?
            .unchecked_into();
        if next.done() {
            return Ok(());
        }
        let chunk: js_sys::Uint8Array = next.value().unchecked_into();
        file.write(&chunk).map_err(Try::Disk)?;
    }
}

/// `fetch` `url`, to its response's headers: an error status is a failure,
/// carrying `Retry-After` if the server exposes it to scripts.
async fn fetch_response(url: &str) -> Result<web_sys::Response, Failed> {
    let init = web_sys::RequestInit::new();
    init.set_method("GET");
    // the body goes to OPFS: keeping it in the HTTP cache too would store
    // every file twice
    init.set_cache(web_sys::RequestCache::NoStore);
    init.set_signal(Some(&web_sys::AbortSignal::timeout_with_u32(
        TIMEOUT.as_millis() as u32,
    )));
    let global = js_sys::global();
    let request = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        window.fetch_with_str_and_init(url, &init)
    } else {
        global
            .unchecked_ref::<web_sys::WorkerGlobalScope>()
            .fetch_with_str_and_init(url, &init)
    };
    // rejects on a network error, a cross-origin refusal or the timeout
    let response: web_sys::Response = JsFuture::from(request)
        .await
        .map_err(js_error)?
        .unchecked_into();
    let status = response.status();
    if status >= 400 {
        let retry_after = response.headers().get("retry-after").ok().flatten();
        return Err(Failed::status(status, retry_after.as_deref()));
    }
    Ok(response)
}

/// A thrown JS exception, as ureq's I/O error (a timeout as `TimedOut`).
fn js_error(e: JsValue) -> ureq::Error {
    let (name, msg) = e.dyn_ref::<js_sys::Error>().map_or_else(
        || (String::new(), format!("{e:?}")),
        |err| (String::from(err.name()), String::from(err.message())),
    );
    let kind = if name == "TimeoutError" {
        std::io::ErrorKind::TimedOut
    } else {
        std::io::ErrorKind::Other
    };
    ureq::Error::Io(std::io::Error::new(kind, msg))
}

/// A thrown JS exception as this crate's error, for the non-HTTP calls.
pub(crate) fn js_failure(what: &str, e: JsValue) -> crate::Error {
    let msg = e
        .dyn_ref::<js_sys::Error>()
        .map_or_else(|| format!("{e:?}"), |err| String::from(err.message()));
    crate::Error::Io(std::io::Error::other(format!("{what}: {msg}")))
}
