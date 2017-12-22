extern crate image;
extern crate rand;

use std::f64::consts::PI;
use std::cmp::min;
use image::{ImageBuffer, Rgb};
use rand::{Rng, ThreadRng};

const W: u32 = 512;
const H: u32 = 512;
const N: u32 = 64;
const MAX_STEP: u32 = 10;
const MAX_DISTANCE: f64 = 2.0;
const EPSILON: f64 = 1e-6;

struct Res {
    sd: f64,
    emissive: f64,
}

fn scene(x: f64, y: f64) -> Res {
    Res {
        sd: circle_sdf(x, y, 0.5, 0.5, 0.1),
        emissive: 2.0,
    }
}

fn circle_sdf(x: f64, y: f64, cx: f64, cy: f64, r: f64) -> f64 {
    let ux = x - cx;
    let uy = y - cy;
    (ux * ux + uy * uy).sqrt() - r
}

fn trace(ox: f64, oy: f64, dx: f64, dy: f64) -> f64 {
    let mut t = 0.0;
    let mut i = 0;
    while i < MAX_STEP && t < MAX_DISTANCE {
        let r = scene(ox + dx * t, oy + dy * t);
        if r.sd < EPSILON {
            return r.emissive;
        }
        i += 1;
        t += r.sd;
    }
    0.0
}

fn sample(rng: &mut ThreadRng, x: f64, y: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        let a = 2.0 * PI * (i as f64 + rng.gen_range(0.0, 1.0)) / N as f64;
        sum += trace(x, y, a.cos(), a.sin());
    }
    sum / N as f64
}

fn main() {
    let mut img = ImageBuffer::from_pixel(W, H, Rgb([0u8, 0u8, 0u8]));
    let mut rng = rand::thread_rng();
    for x in 0..W {
        for y in 0..H {
            let xx = x as f64 / W as f64;
            let yy = y as f64 / H as f64;
            let brightness = min((sample(&mut rng, xx, yy) * 255.0) as u32, 255) as u8;
            img.put_pixel(x, y, Rgb([brightness, brightness, brightness]));
        }
    }
    img.save("out.png").unwrap();
}
