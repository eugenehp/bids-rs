//! `collect_schema` over https (the `http` feature), natively; in a browser,
//! `browser.rs`. The ones that reach the schema host are behind `#[ignore]`:
//! `cargo test -p bids-schema --features http --test collect_http -- --ignored`.

#![cfg(all(
    feature = "http",
    not(all(target_family = "wasm", target_os = "unknown"))
))]

use bids_schema::collect::{SchemaSource, collect_schema};

#[test]
fn an_unreachable_endpoint_is_an_error() {
    // nothing listens on port 9 (discard)
    let e = collect_schema(SchemaSource::Url("http://127.0.0.1:9/schema.json")).unwrap_err();
    assert!(e.to_string().contains("not reachable"), "{e}");
}

#[test]
#[ignore] // requires network
fn fetches_a_published_schema() {
    let schema = collect_schema(SchemaSource::Version("1.10.0")).unwrap();
    assert_eq!(schema["bids_version"], "1.10.0");
    assert!(schema["objects"]["entities"].is_object());
}
