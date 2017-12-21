extern crate image;

use image::{ImageBuffer, Rgb};

const W: u32 = 512;
const H: u32 = 512;

fn main() {
    let img = ImageBuffer::from_pixel(W, H, Rgb([0u8, 0u8, 0u8]));
    img.save("out.png").unwrap();
}
