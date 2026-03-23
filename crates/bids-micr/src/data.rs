//! Microscopy image reading via the TIFF/OME-TIFF format.
//!
//! Requires the `tiff` feature flag: `bids-micr = { features = ["tiff"] }`
//!
//! Reads TIFF and OME-TIFF microscopy images into memory as pixel arrays.
//! Supports common sample types: 8-bit, 16-bit, and 32-bit grayscale/RGB.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// A decoded microscopy image.
#[derive(Debug, Clone)]
pub struct MicrImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of channels (1 = grayscale, 3 = RGB, 4 = RGBA).
    pub channels: usize,
    /// Bits per sample per channel.
    pub bits_per_sample: u16,
    /// Pixel data as f64 values, normalized to [0, 1] for integer types.
    /// Layout: row-major, interleaved channels: [r,g,b, r,g,b, ...] for RGB.
    pub data: Vec<f64>,
    /// Number of IFDs (pages/slices) in the TIFF file.
    pub n_pages: usize,
}

impl MicrImage {
    /// Total number of pixels (width × height).
    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Get a single pixel value at (x, y) for single-channel images.
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<f64> {
        if x >= self.width || y >= self.height { return None; }
        let idx = (y as usize * self.width as usize + x as usize) * self.channels;
        self.data.get(idx).copied()
    }

    /// Get all channel values at (x, y).
    pub fn get_pixel_channels(&self, x: u32, y: u32) -> Option<Vec<f64>> {
        if x >= self.width || y >= self.height { return None; }
        let idx = (y as usize * self.width as usize + x as usize) * self.channels;
        if idx + self.channels <= self.data.len() {
            Some(self.data[idx..idx + self.channels].to_vec())
        } else {
            None
        }
    }

    /// Compute the mean pixel intensity (across all channels and pixels).
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }
}

/// Read a microscopy image from a TIFF or OME-TIFF file.
///
/// Reads the first IFD (page) and converts pixel data to f64. Integer pixel
/// values are normalized to [0.0, 1.0]; float pixels are kept as-is.
///
/// Supports: 8-bit, 16-bit, 32-bit grayscale and RGB images.
pub fn read_tiff(path: &Path) -> Result<MicrImage> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::ColorType;

    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::Other, format!("TIFF error: {}", e)
        )))?;

    let (width, height) = decoder.dimensions()
        .map_err(|e| BidsError::Csv(format!("TIFF dimensions error: {}", e)))?;

    let color_type = decoder.colortype()
        .map_err(|e| BidsError::Csv(format!("TIFF colortype error: {}", e)))?;

    let (channels, bits_per_sample) = match color_type {
        ColorType::Gray(bps) => (1, bps as u16),
        ColorType::RGB(bps) => (3, bps as u16),
        ColorType::RGBA(bps) => (4, bps as u16),
        ColorType::GrayA(bps) => (2, bps as u16),
        _ => return Err(BidsError::FileType(format!("Unsupported TIFF color type: {:?}", color_type))),
    };

    let image = decoder.read_image()
        .map_err(|e| BidsError::Csv(format!("TIFF read error: {}", e)))?;

    let data: Vec<f64> = match image {
        DecodingResult::U8(buf) => {
            buf.iter().map(|&v| v as f64 / 255.0).collect()
        }
        DecodingResult::U16(buf) => {
            buf.iter().map(|&v| v as f64 / 65535.0).collect()
        }
        DecodingResult::U32(buf) => {
            buf.iter().map(|&v| v as f64 / 4294967295.0).collect()
        }
        DecodingResult::F32(buf) => {
            buf.iter().map(|&v| v as f64).collect()
        }
        DecodingResult::F64(buf) => {
            buf.to_vec()
        }
        _ => return Err(BidsError::FileType("Unsupported TIFF sample format".into())),
    };

    // Count pages by iterating IFDs
    let mut n_pages = 1;
    while decoder.more_images() {
        if decoder.next_image().is_ok() {
            n_pages += 1;
        } else {
            break;
        }
    }

    Ok(MicrImage {
        width, height, channels, bits_per_sample, data, n_pages,
    })
}

/// Read all pages (slices) from a multi-page TIFF as separate images.
pub fn read_tiff_stack(path: &Path) -> Result<Vec<MicrImage>> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::ColorType;

    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::Other, format!("TIFF error: {}", e)
        )))?;

    let mut pages = Vec::new();

    loop {
        let (width, height) = decoder.dimensions()
            .map_err(|e| BidsError::Csv(format!("TIFF dimensions error: {}", e)))?;

        let color_type = decoder.colortype()
            .map_err(|e| BidsError::Csv(format!("TIFF colortype error: {}", e)))?;

        let (channels, bits_per_sample) = match color_type {
            ColorType::Gray(bps) => (1, bps as u16),
            ColorType::RGB(bps) => (3, bps as u16),
            ColorType::RGBA(bps) => (4, bps as u16),
            ColorType::GrayA(bps) => (2, bps as u16),
            _ => return Err(BidsError::FileType(format!("Unsupported TIFF color type: {:?}", color_type))),
        };

        let image = decoder.read_image()
            .map_err(|e| BidsError::Csv(format!("TIFF read error: {}", e)))?;

        let data: Vec<f64> = match image {
            DecodingResult::U8(buf) => buf.iter().map(|&v| v as f64 / 255.0).collect(),
            DecodingResult::U16(buf) => buf.iter().map(|&v| v as f64 / 65535.0).collect(),
            DecodingResult::U32(buf) => buf.iter().map(|&v| v as f64 / 4294967295.0).collect(),
            DecodingResult::F32(buf) => buf.iter().map(|&v| v as f64).collect(),
            DecodingResult::F64(buf) => buf.to_vec(),
            _ => return Err(BidsError::FileType("Unsupported TIFF sample format".into())),
        };

        pages.push(MicrImage {
            width, height, channels, bits_per_sample, data, n_pages: 0,
        });

        if decoder.more_images() {
            if decoder.next_image().is_err() { break; }
        } else {
            break;
        }
    }

    // Set n_pages on all entries
    let total = pages.len();
    for p in &mut pages { p.n_pages = total; }

    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Create a minimal valid TIFF file for testing.
    fn create_test_tiff(path: &Path, width: u32, height: u32) {
        use tiff::encoder::TiffEncoder;
        use tiff::encoder::colortype::Gray8;

        let file = std::fs::File::create(path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        let mut encoder = TiffEncoder::new(&mut writer).unwrap();

        let pixels: Vec<u8> = (0..(width * height) as usize)
            .map(|i| (i % 256) as u8)
            .collect();

        encoder.write_image::<Gray8>(width, height, &pixels).unwrap();
    }

    #[test]
    fn test_read_tiff_grayscale() {
        let dir = std::env::temp_dir().join("bids_micr_tiff_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.tif");
        create_test_tiff(&path, 16, 8);

        let img = read_tiff(&path).unwrap();
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 8);
        assert_eq!(img.channels, 1);
        assert_eq!(img.data.len(), 128); // 16 × 8
        assert_eq!(img.bits_per_sample, 8);
        // First pixel: 0/255 = 0.0
        assert!((img.data[0] - 0.0).abs() < 0.01);
        // Second pixel: 1/255
        assert!((img.data[1] - 1.0 / 255.0).abs() < 0.001);

        assert!((img.get_pixel(0, 0).unwrap() - 0.0).abs() < 0.01);
        assert!(img.mean() > 0.0);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
