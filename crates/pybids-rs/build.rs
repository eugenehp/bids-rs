//! Link configuration for the Python extension module.

fn main() {
    // A PyO3 extension module deliberately leaves the CPython symbols
    // (`_PyList_Type`, `__Py_Dealloc`, …) unresolved: the interpreter supplies
    // them when it dlopen's the module. macOS' linker rejects that by default,
    // so the cdylib has to opt in.
    //
    // maturin passes these flags itself, which is why `maturin build` works.
    // Plain `cargo build` and the verification build inside `cargo publish`
    // do not, and fail to link without this.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
