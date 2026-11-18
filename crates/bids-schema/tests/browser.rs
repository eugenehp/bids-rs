//! `collect_schema` over https on browser wasm, where the browser's
//! XMLHttpRequest fetches it: in a dedicated Web Worker, as `bids-dataset`'s
//! browser tests (see there for running them), with `--no-default-features
//! --features http,tls-rustcrypto`. The one that reaches the schema host is
//! behind `#[ignore]`: `-- --include-ignored`.

#![cfg(all(feature = "http", target_family = "wasm", target_os = "unknown"))]

use bids_schema::collect::{SchemaSource, collect_schema};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_dedicated_worker);

#[wasm_bindgen_test]
fn an_unreachable_endpoint_is_an_error_not_a_panic() {
    // nothing listens on port 9 (discard)
    let e = collect_schema(SchemaSource::Url("http://127.0.0.1:9/schema.json")).unwrap_err();
    assert!(e.to_string().contains("not reachable"), "{e}");
}

#[wasm_bindgen_test]
fn a_path_is_an_error_without_a_filesystem() {
    assert!(collect_schema(SchemaSource::Path(std::path::Path::new("schema.json"))).is_err());
}

#[wasm_bindgen_test]
#[ignore]
fn fetches_a_published_schema() {
    let schema = collect_schema(SchemaSource::Version("1.10.0")).unwrap();
    assert_eq!(schema["bids_version"], "1.10.0");
    assert!(schema["objects"]["entities"].is_object());
}
