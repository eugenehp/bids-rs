//! The browser's Origin Private File System (OPFS): a private disk per site
//! that persists across visits, where browser wasm has no other.
//! [`OpenNeuro::download_to_opfs`](crate::OpenNeuro::download_to_opfs)
//! downloads into it, and [`read`] reads a file back.
//!
//! Files are written with synchronous access handles, which every browser
//! with OPFS has (Safari's only way to write it) but only in a dedicated Web
//! Worker: elsewhere these fail, saying so. Paths are `/`-separated and
//! relative to the OPFS root, e.g. `ds000001/sub-01/anat/sub-01_T1w.nii.gz`.

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetDirectoryOptions,
    FileSystemGetFileOptions, FileSystemHandle, FileSystemHandleKind, FileSystemRemoveOptions,
    FileSystemSyncAccessHandle,
};

use crate::web::js_failure;

/// Read the file at `path` (relative to the OPFS root).
pub async fn read(path: &str) -> crate::Result<Vec<u8>> {
    let (dir, name) = parent(&root().await?, path, false).await?;
    let file = file_handle(&dir, name, false).await?;
    let blob: web_sys::File = JsFuture::from(file.get_file())
        .await
        .map_err(|e| js_failure(path, e))?
        .unchecked_into();
    let buffer = JsFuture::from(blob.array_buffer())
        .await
        .map_err(|e| js_failure(path, e))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// `path` as an OPFS path (`/`-separated, as wasm's paths are).
pub(crate) fn path_str(path: &std::path::Path) -> crate::Result<&str> {
    path.to_str().ok_or_else(|| {
        crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("not a UTF-8 path: {}", path.display()),
        ))
    })
}

/// Fail unless this is a dedicated worker, the one place sync access handles
/// (so [`write`]) exist.
pub(crate) fn require_dedicated_worker() -> crate::Result<()> {
    if js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
    {
        Ok(())
    } else {
        Err(crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "OPFS downloads write with synchronous access handles, which only a \
             dedicated Web Worker has: run them in one (`new Worker(...)`)",
        )))
    }
}

/// The OPFS root directory, `navigator.storage.getDirectory()`.
pub(crate) async fn root() -> crate::Result<FileSystemDirectoryHandle> {
    let global = js_sys::global();
    let storage = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        window.navigator().storage()
    } else {
        global
            .unchecked_ref::<web_sys::WorkerGlobalScope>()
            .navigator()
            .storage()
    };
    Ok(JsFuture::from(storage.get_directory())
        .await
        .map_err(|e| js_failure("OPFS root", e))?
        .unchecked_into())
}

/// The directory `path` names under `dir`, created (with its parents) if
/// asked to.
pub(crate) async fn dir(
    dir: &FileSystemDirectoryHandle,
    path: &str,
    create: bool,
) -> crate::Result<FileSystemDirectoryHandle> {
    let options = FileSystemGetDirectoryOptions::new();
    options.set_create(create);
    let mut current = dir.clone();
    for part in path.split('/').filter(|p| !p.is_empty()) {
        current = JsFuture::from(current.get_directory_handle_with_options(part, &options))
            .await
            .map_err(|e| js_failure(path, e))?
            .unchecked_into();
    }
    Ok(current)
}

/// The size of the file at `path` under `dir`, if there is one.
pub(crate) async fn size(dir: &FileSystemDirectoryHandle, path: &str) -> Option<u64> {
    let (parent_dir, name) = parent(dir, path, false).await.ok()?;
    let file = file_handle(&parent_dir, name, false).await.ok()?;
    let blob: web_sys::File = JsFuture::from(file.get_file()).await.ok()?.unchecked_into();
    Some(blob.size() as u64)
}

/// Write `data` as the file at `path` under `dir`, creating its directories
/// and replacing what was there. A failed write removes the file, so a
/// resumed download doesn't take it for finished.
pub(crate) async fn write(
    dir: &FileSystemDirectoryHandle,
    path: &str,
    data: &[u8],
) -> crate::Result<()> {
    let mut file = Writer::create(dir, path).await?;
    match file.write_bytes(data) {
        Ok(()) => file.finish().await.map(|_| ()),
        Err(e) => {
            file.abandon().await;
            Err(e)
        }
    }
}

/// A file being written: replaced from the start, then appended to chunk by
/// chunk, through a synchronous access handle (so only in a dedicated
/// worker), which is the file's lock until closed. [`finish`](Self::finish)
/// keeps it, [`abandon`](Self::abandon) removes it; dropped unfinished (a
/// cancelled download), it's closed and left for the next run to replace.
pub(crate) struct Writer {
    handle: FileSystemSyncAccessHandle,
    parent: FileSystemDirectoryHandle,
    name: String,
    path: String,
    written: u64,
}

impl Writer {
    /// Open the file at `path` under `dir` (creating its directories) and
    /// empty it.
    pub(crate) async fn create(dir: &FileSystemDirectoryHandle, path: &str) -> crate::Result<Self> {
        // outside a dedicated worker `createSyncAccessHandle` doesn't exist,
        // and calling it would throw past this function
        require_dedicated_worker()?;
        let (parent, name) = parent(dir, path, true).await?;
        let file = file_handle(&parent, name, true).await?;
        let handle: FileSystemSyncAccessHandle = JsFuture::from(file.create_sync_access_handle())
            .await
            .map_err(|e| js_failure(path, e))?
            .unchecked_into();
        let writer = Self {
            handle,
            parent,
            name: name.to_string(),
            path: path.to_string(),
            written: 0,
        };
        writer
            .handle
            .truncate_with_u32(0)
            .map_err(|e| js_failure(path, e))?;
        Ok(writer)
    }

    /// Append a chunk straight from JS memory (a response body's), without
    /// copying it into wasm's.
    pub(crate) fn write(&mut self, chunk: &js_sys::Uint8Array) -> crate::Result<()> {
        let n = self
            .handle
            .write_with_js_u8_array(chunk)
            .map_err(|e| js_failure(&self.path, e))?;
        self.wrote(n, chunk.length() as usize)
    }

    /// Append bytes from wasm memory.
    pub(crate) fn write_bytes(&mut self, data: &[u8]) -> crate::Result<()> {
        let n = self
            .handle
            .write_with_u8_array(data)
            .map_err(|e| js_failure(&self.path, e))?;
        self.wrote(n, data.len())
    }

    /// The bytes written so far.
    pub(crate) fn written(&self) -> u64 {
        self.written
    }

    fn wrote(&mut self, n: f64, len: usize) -> crate::Result<()> {
        self.written += n as u64;
        // a short write is a full disk (the quota)
        if n as usize == len {
            Ok(())
        } else {
            Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                format!("{}: wrote {n} of {len} bytes (storage quota?)", self.path),
            )))
        }
    }

    /// Flush and close, keeping the file; returns its size. If the flush
    /// fails the file is removed.
    pub(crate) async fn finish(self) -> crate::Result<u64> {
        match self.handle.flush() {
            Ok(()) => Ok(self.written),
            Err(e) => {
                let err = js_failure(&self.path, e);
                self.abandon().await;
                Err(err)
            }
        }
    }

    /// Close and remove the file.
    pub(crate) async fn abandon(self) {
        self.handle.close();
        let _ = JsFuture::from(self.parent.remove_entry(&self.name)).await;
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // releases the file's lock; closing twice is harmless
        self.handle.close();
    }
}

/// The directory at `path` under `dir`, if there is one.
pub(crate) async fn try_dir(
    dir: &FileSystemDirectoryHandle,
    path: &str,
) -> Option<FileSystemDirectoryHandle> {
    self::dir(dir, path, false).await.ok()
}

/// `dir`'s entries: name, and whether it's a directory.
pub(crate) async fn list(dir: &FileSystemDirectoryHandle) -> crate::Result<Vec<(String, bool)>> {
    Ok(handles(dir)
        .await?
        .into_iter()
        .map(|h| (h.name(), h.kind() == FileSystemHandleKind::Directory))
        .collect())
}

/// The total size of the files under `dir`, at any depth.
pub(crate) async fn tree_size(dir: &FileSystemDirectoryHandle) -> u64 {
    let mut total = 0;
    let mut pending = vec![dir.clone()];
    while let Some(dir) = pending.pop() {
        for handle in handles(&dir).await.unwrap_or_default() {
            if handle.kind() == FileSystemHandleKind::Directory {
                pending.push(handle.unchecked_into());
            } else if let Ok(file) =
                JsFuture::from(handle.unchecked_ref::<FileSystemFileHandle>().get_file()).await
            {
                total += file.unchecked_into::<web_sys::File>().size() as u64;
            }
        }
    }
    total
}

/// Remove the entry at `path` under `dir` (a directory with all it holds).
/// Nothing there is success, as `std::fs` callers check first.
pub(crate) async fn remove(dir: &FileSystemDirectoryHandle, path: &str) -> crate::Result<()> {
    let (dirs, name) = path
        .trim_end_matches('/')
        .rsplit_once('/')
        .unwrap_or(("", path));
    if name.is_empty() {
        // the root itself: empty it
        for (entry, _) in list(dir).await? {
            Box::pin(remove(dir, &entry)).await?;
        }
        return Ok(());
    }
    let Some(parent_dir) = try_dir(dir, dirs).await else {
        return Ok(());
    };
    let options = FileSystemRemoveOptions::new();
    options.set_recursive(true);
    match JsFuture::from(parent_dir.remove_entry_with_options(name, &options)).await {
        Ok(_) => Ok(()),
        Err(e) if js_name(&e) == "NotFoundError" => Ok(()),
        Err(e) => Err(js_failure(path, e)),
    }
}

/// `dir`'s entries' handles.
async fn handles(dir: &FileSystemDirectoryHandle) -> crate::Result<Vec<FileSystemHandle>> {
    let entries = dir.values();
    let mut out = Vec::new();
    loop {
        let next = entries.next().map_err(|e| js_failure("OPFS listing", e))?;
        let next: js_sys::IteratorNext = JsFuture::from(next)
            .await
            .map_err(|e| js_failure("OPFS listing", e))?
            .unchecked_into();
        if next.done() {
            return Ok(out);
        }
        out.push(next.value().unchecked_into());
    }
}

/// A thrown DOMException's name (`NotFoundError`, ...).
fn js_name(e: &wasm_bindgen::JsValue) -> String {
    e.dyn_ref::<js_sys::Error>()
        .map(|err| String::from(err.name()))
        .unwrap_or_default()
}

/// `path`'s directory under `dir` and its file name.
async fn parent<'p>(
    dir: &FileSystemDirectoryHandle,
    path: &'p str,
    create: bool,
) -> crate::Result<(FileSystemDirectoryHandle, &'p str)> {
    let (dirs, name) = path.rsplit_once('/').unwrap_or(("", path));
    Ok((self::dir(dir, dirs, create).await?, name))
}

async fn file_handle(
    dir: &FileSystemDirectoryHandle,
    name: &str,
    create: bool,
) -> crate::Result<FileSystemFileHandle> {
    let options = FileSystemGetFileOptions::new();
    options.set_create(create);
    Ok(
        JsFuture::from(dir.get_file_handle_with_options(name, &options))
            .await
            .map_err(|e| js_failure(name, e))?
            .unchecked_into(),
    )
}
