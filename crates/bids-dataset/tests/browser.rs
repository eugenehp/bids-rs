//! Browser wasm tests, in a dedicated Web Worker (where OPFS can be written):
//! `cargo test -p bids-dataset --target wasm32-unknown-unknown`, with
//! `wasm-bindgen-test-runner` (`cargo install wasm-bindgen-cli`, the version in
//! Cargo.lock) and a WebDriver: `CHROMEDRIVER=...` (or `GECKODRIVER`,
//! `SAFARIDRIVER`).
//!
//! The ones that reach OpenNeuro are behind `#[ignore]`, as in
//! `live_openneuro.rs`: `-- --include-ignored` runs them too.

#![cfg(all(target_family = "wasm", target_os = "unknown"))]

use bids_dataset::ratelimit::RateLimiter;
use bids_dataset::{Cache, OpenNeuro, RemoteFile, opfs};
use std::path::PathBuf;
use std::time::Duration;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_dedicated_worker);

/// A cache of its own per test, as tests share the origin's OPFS.
async fn fresh_cache(name: &str) -> Cache {
    let cache = Cache::new(PathBuf::from(format!("bids-rs-test/{name}")));
    cache.clean().await.unwrap();
    cache
}

#[wasm_bindgen_test]
async fn cache_lifecycle() {
    let c = fresh_cache("lifecycle").await;
    assert_eq!(c.total_size().await, 0);
    assert!(c.list_datasets().await.is_empty());
    assert_eq!(c.resolve("ds000001", None).await, None);

    let f = c
        .store_file("ds000001", "1.0.0", "sub-01/anat/a.txt", b"one")
        .await
        .unwrap();
    assert_eq!(
        f,
        c.snapshot_dir("ds000001", "1.0.0")
            .join("sub-01/anat/a.txt")
    );
    assert!(!c.is_cached("ds000001", "1.0.0").await);
    c.mark_complete("ds000001", "1.0.0").await.unwrap();
    c.store_file("ds000001", "2.0.0", "b.txt", b"second")
        .await
        .unwrap();
    c.mark_complete("ds000001", "2.0.0").await.unwrap();
    c.store_file("ds000002", "1.0.0", "c.txt", b"x")
        .await
        .unwrap();

    assert!(c.is_cached("ds000001", "1.0.0").await);
    assert!(c.is_cached("ds000001", "2.0.0").await);
    assert!(!c.is_cached("ds000001", "3.0.0").await);
    assert!(!c.is_cached("ds000002", "1.0.0").await);

    // latest is the last marked complete
    assert_eq!(
        c.resolve("ds000001", None).await,
        Some(c.snapshot_dir("ds000001", "2.0.0"))
    );
    assert_eq!(
        c.resolve("ds000001", Some("1.0.0")).await,
        Some(c.snapshot_dir("ds000001", "1.0.0"))
    );
    assert_eq!(c.resolve("ds000001", Some("9.9.9")).await, None);
    assert_eq!(c.list_versions("ds000001").await, ["1.0.0", "2.0.0"]);

    let datasets = c.list_datasets().await;
    let ids: Vec<_> = datasets.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, ["ds000001", "ds000002"]);
    // ds000001: "one" + "second" + two `.complete` and three refs of 5 bytes
    assert_eq!(datasets[0].size_bytes, 3 + 6 + 2 * 5 + 3 * 5);
    assert_eq!(c.total_size().await, 3 + 6 + 2 * 5 + 3 * 5 + 1);

    c.evict("ds000001", "2.0.0").await.unwrap();
    assert_eq!(c.list_versions("ds000001").await, ["1.0.0"]);
    // nothing there: fine, as natively
    c.evict("ds000001", "7.7.7").await.unwrap();
    c.evict_dataset("ds000002").await.unwrap();
    c.evict_dataset("ds999999").await.unwrap();
    let ids: Vec<_> = c.list_datasets().await.into_iter().map(|d| d.id).collect();
    assert_eq!(ids, ["ds000001"]);

    c.clean().await.unwrap();
    assert_eq!(c.total_size().await, 0);
    c.clean().await.unwrap();
}

#[wasm_bindgen_test]
async fn opfs_round_trip_is_byte_exact() {
    let c = fresh_cache("bytes").await;
    // every byte value, at every offset mod 256, and an empty file
    let data: Vec<u8> = (0..70_000u32).map(|i| (i * 7 % 256) as u8).collect();
    let path = c.store_file("ds", "v", "all.bin", &data).await.unwrap();
    assert_eq!(opfs::read(path.to_str().unwrap()).await.unwrap(), data);
    // rewriting replaces, never appends
    c.store_file("ds", "v", "all.bin", b"short").await.unwrap();
    assert_eq!(opfs::read(path.to_str().unwrap()).await.unwrap(), b"short");
    let empty = c.store_file("ds", "v", "empty", b"").await.unwrap();
    assert!(
        opfs::read(empty.to_str().unwrap())
            .await
            .unwrap()
            .is_empty()
    );
    assert!(opfs::read("bids-rs-test/bytes/no/such/file").await.is_err());
    c.clean().await.unwrap();
}

#[wasm_bindgen_test]
fn rate_limiter_waits() {
    // one request per 0.5 s: the second waits that long, Atomics or spinning
    let limiter = RateLimiter::new(2.0);
    limiter.acquire();
    let start = web_time::Instant::now();
    limiter.acquire();
    let waited = start.elapsed().as_millis();
    assert!((450..=1500).contains(&waited), "waited {waited} ms");
}

#[wasm_bindgen_test]
fn retry_after_dates_read_the_browser_clock() {
    // std's clock panics here: this is web-time's
    let past = bids_dataset::ratelimit::parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT");
    assert_eq!(past, Some(Duration::ZERO));
    let future = bids_dataset::ratelimit::parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT");
    assert!(future.unwrap() > Duration::from_secs(365 * 86_400));
}

#[wasm_bindgen_test]
fn unreachable_host_is_an_error_not_a_panic() {
    let agent = bids_dataset::http::make_agent();
    // nothing listens on port 9 (discard)
    let r = bids_dataset::http::get_with_retry(&agent, "http://127.0.0.1:9/", 0);
    assert!(matches!(r, Err(bids_dataset::Error::Network(_))), "{r:?}");
}

#[wasm_bindgen_test]
fn download_dataset_points_at_the_browser_ways() {
    let r = OpenNeuro::new().download_dataset(
        "ds000001",
        std::path::Path::new("x"),
        None::<fn(&RemoteFile) -> bool>,
    );
    let e = r.unwrap_err().to_string();
    assert!(
        e.contains("download_to_opfs") && e.contains("fetch_file"),
        "{e}"
    );
}

// --- over the network (OpenNeuro) ---

fn small(f: &RemoteFile) -> bool {
    f.path == "dataset_description.json" || f.path.starts_with("sub-01/anat/")
}

#[wasm_bindgen_test]
#[ignore]
fn live_list_and_fetch_binary() {
    let on = OpenNeuro::new();
    let files = on.list_files("ds000001", None).unwrap();
    assert!(files.len() > 100, "{} files", files.len());
    let nii = files
        .iter()
        .filter(|f| f.path.ends_with(".nii.gz"))
        .min_by_key(|f| f.size)
        .unwrap();
    let data = on.fetch_file("ds000001", &nii.path).unwrap();
    assert_eq!(data.len() as u64, nii.size);
    assert_eq!(&data[..2], [0x1f, 0x8b], "gzip magic");
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&data[..]), &mut raw).unwrap();
    // a NIfTI-1 header: sizeof_hdr 348
    assert_eq!(i32::from_le_bytes(raw[..4].try_into().unwrap()), 348);
}

#[wasm_bindgen_test]
#[ignore]
fn live_search() {
    let hits = OpenNeuro::new()
        .search()
        .modality("eeg")
        .limit(3)
        .execute()
        .unwrap();
    assert!(!hits.is_empty());
}

#[wasm_bindgen_test]
#[ignore]
async fn live_download_to_cache_and_resume() {
    let c = fresh_cache("live").await;
    let on = OpenNeuro::new();
    let snap = on
        .download_to_cache("ds000001", "1.0.0", &c, Some(small))
        .await
        .unwrap();
    assert!(c.is_cached("ds000001", "1.0.0").await);
    let listed: Vec<_> = on
        .list_files("ds000001", None)
        .unwrap()
        .into_iter()
        .filter(small)
        .collect();
    for rf in &listed {
        let cached = opfs::read(snap.join(&rf.path).to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(cached.len() as u64, rf.size, "{}", rf.path);
    }

    // complete: returns without downloading; after an evict, downloads again
    let again = on
        .download_to_cache("ds000001", "1.0.0", &c, Some(small))
        .await
        .unwrap();
    assert_eq!(again, snap);
    c.evict("ds000001", "1.0.0").await.unwrap();
    on.download_to_cache("ds000001", "1.0.0", &c, Some(small))
        .await
        .unwrap();
    assert!(c.is_cached("ds000001", "1.0.0").await);
    c.clean().await.unwrap();
}

/// wasm's linear memory, which only ever grows.
fn wasm_memory() -> f64 {
    wasm_bindgen::memory()
        .unchecked_into::<js_sys::WebAssembly::Memory>()
        .buffer()
        .unchecked_into::<js_sys::ArrayBuffer>()
        .byte_length() as f64
}

#[wasm_bindgen_test]
#[ignore]
async fn live_large_file_streams_without_growing_memory() {
    let on = OpenNeuro::new();
    // the largest, ~50 MB: bigger than anything the tests before hold
    let largest = on
        .list_files("ds000001", None)
        .unwrap()
        .into_iter()
        .max_by_key(|f| f.size)
        .unwrap();
    assert!(largest.size > 30_000_000, "{largest:?}");
    let path = largest.path.clone();

    // a cleaned cache: downloaded for certain, not skipped as already there
    let c = fresh_cache("large").await;
    let before = wasm_memory();
    let snap = on
        .download_to_cache(
            "ds000001",
            "1.0.0",
            &c,
            Some(|f: &RemoteFile| f.path == path),
        )
        .await
        .unwrap();
    let grew = wasm_memory() - before;
    // streamed: chunks pass from the response to the file in JS, the file
    // never in wasm memory
    assert!(
        grew < 8e6,
        "wasm memory grew {grew} bytes for a {} byte file",
        largest.size
    );

    let stored = opfs::read(snap.join(&largest.path).to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(stored.len() as u64, largest.size);
    assert_eq!(stored, on.fetch_file("ds000001", &largest.path).unwrap());
    c.clean().await.unwrap();
}

#[wasm_bindgen_test]
#[ignore]
async fn live_download_to_opfs_resumes() {
    let on = OpenNeuro::new();
    let first = on.download_to_opfs("ds000001", Some(small)).await.unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(first.downloaded + first.skipped, first.total_files);
    let second = on.download_to_opfs("ds000001", Some(small)).await.unwrap();
    assert_eq!(second.skipped, second.total_files);
    assert_eq!(second.downloaded, 0);
}
