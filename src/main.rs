extern crate image;
extern crate rand;
extern crate rayon;

use std::f64::consts::PI;
use std::cmp::min;
use image::{ImageBuffer, Rgb};
use rand::{Rng, ThreadRng};
use rayon::prelude::*;

const W: u32 = 512;
const H: u32 = 512;
const N: u32 = 256;
const MAX_STEP: u32 = 64;
const MAX_DISTANCE: f64 = 2.0;
const EPSILON: f64 = 1e-6;
const BIAS: f64 = 1e-4;
const MAX_DEPTH: u32 = 3;

struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn black() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }
}

impl std::ops::Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Color {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl std::ops::Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Color {
        Color {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl std::ops::Mul<f64> for Color {
    type Output = Color;

    fn mul(self, s: f64) -> Color {
        Color {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }
}

impl std::iter::Sum for Color {
    fn sum<I>(iter: I) -> Color
        where I: Iterator<Item = Color>
    {
        iter.fold(Color::black(), std::ops::Add::add)
    }
}

struct Res {
    sd: f64,
    emissive: Color,
    reflectivity: f64,
    eta: f64,
    absorption: Color,
}

impl std::ops::Add<Res> for Res {
    type Output = Res;

    fn add(self, rhs: Res) -> Res {
        if self.sd < rhs.sd {
            self
        } else {
            rhs
        }
    }
}

impl std::ops::Sub<Res> for Res {
    type Output = Res;

    fn sub(self, rhs: Res) -> Res {
        Res {
            sd: if self.sd > -rhs.sd {
                self.sd
            } else {
                -rhs.sd
            },
            .. self
        }
    }
}

impl std::ops::Mul<Res> for Res {
    type Output = Res;

    fn mul(self, rhs: Res) -> Res {
        if self.sd > rhs.sd {
            self
        } else {
            rhs
        }
    }
}


fn scene(x: f64, y: f64) -> Res {
    let a = Res { 
        sd: circle_sdf(x, y, 0.5, -0.2, 0.1),
        emissive: Color {
            r: 10.0,
            g: 10.0,
            b: 10.0,
        },
        reflectivity: 0.0,
        eta: 0.0,
        absorption: Color::black(),
    };
    let b = Res {
        sd: ngon_sdf(x, y, 0.5, 0.5, 0.25, 5.0),
        emissive: Color::black(),
        reflectivity: 0.0,
        eta: 1.5,
        absorption: Color {
            r: 4.0,
            g: 4.0,
            b: 1.0,
        },
    };
    a + b
}

fn circle_sdf(x: f64, y: f64, cx: f64, cy: f64, r: f64) -> f64 {
    let ux = x - cx;
    let uy = y - cy;
    (ux * ux + uy * uy).sqrt() - r
}

fn plane_sdf(x: f64, y: f64, px: f64, py: f64, nx: f64, ny: f64) -> f64 {
    (x - px) * nx + (y - py) * ny
}

fn segment_sdf(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let vx = x - ax;
    let vy = y - ay;
    let ux = bx - ax;
    let uy = by - ay;
    let t = ((vx * ux + vy * uy) / (ux * ux + uy * uy)).min(1.0).max(0.0);
    let dx = vx - ux * t;
    let dy = vy - uy * t;
    (dx * dx + dy * dy).sqrt()
}

fn capsule_sdf(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64, r: f64) -> f64 {
    segment_sdf(x, y, ax, ay, bx, by) - r
}

fn box_sdf(x: f64, y: f64, cx: f64, cy: f64, theta: f64, sx: f64, sy: f64) -> f64 {
    let costheta = theta.cos();
    let sintheta = theta.sin();
    let dx = ((x - cx) * costheta + (y - cy) * sintheta).abs() - sx;
    let dy = ((y - cy) * costheta - (x - cx) * sintheta).abs() - sy;
    let ax = dx.max(0.0);
    let ay = dy.max(0.0);
    dx.max(dy).min(0.0) + (ax * ax + ay * ay).sqrt()
}

fn triangle_sdf(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    let d = segment_sdf(x, y, ax, ay, bx, by)
        .min(segment_sdf(x, y, bx, by, cx, cy))
        .min(segment_sdf(x, y, cx, cy, ax, ay));
    if (bx - ax) * (y - ay) > (by - ay) * (x - ax) &&
        (cx - bx) * (y - by) > (cy - by) * (x - bx) &&
        (ax - cx) * (y - cy) > (ay - cy) * (x - cx) {
        -d
    } else {
        d
    }
}

fn ngon_sdf(x: f64, y: f64, cx: f64, cy: f64, r: f64, n: f64) -> f64 {
    let ux = x - cx;
    let uy = y - cy;
    let a = 2.0 * PI / n;
    let t = (uy.atan2(ux) + 2.0 * PI) % a;
    let s = (ux * ux + uy * uy).sqrt();
    plane_sdf(s * t.cos(), s * t.sin(), r, 0.0, (a * 0.5).cos(), (a * 0.5).sin())
}

fn reflect(ix: f64, iy: f64, nx: f64, ny: f64) -> (f64, f64) {
    let dot2 = (ix * nx + iy * ny) * 2.0;
    (ix - dot2 * nx, iy - dot2 * ny)
}

fn gradient(x: f64, y: f64) -> (f64, f64) {
    let nx = (scene(x + EPSILON, y).sd - scene(x - EPSILON, y).sd) * (0.5 / EPSILON);
    let ny = (scene(x, y + EPSILON).sd - scene(x, y - EPSILON).sd) * (0.5 / EPSILON);
    (nx, ny)
}

fn refract(ix: f64, iy: f64, nx: f64, ny: f64, eta: f64) -> Option<(f64, f64)> {
    let dot = ix * nx + iy * ny;
    let k = 1.0 - eta * eta * (1.0 - dot * dot);
    if k < 0.0 {
        return None; // all reflection
    }
    let a = eta * dot + k.sqrt();
    Some((eta * ix - a * nx, eta * iy - a * ny))
}

fn fresnel(cosi: f64, cost: f64, etai: f64, etat: f64) -> f64 {
    let rs = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
    let rp = (etat * cost - etai * cosi) / (etat * cost + etai * cosi);
    (rs * rs + rp * rp) * 0.5
}

fn schlick(cosi: f64, cost: f64, etai: f64, etat: f64) -> f64 {
    let r0 = (etai - etat) / (etai + etat);
    let r0 = r0 * r0;
    let a = if etai < etat {
        1.0 - cosi
    } else {
        1.0 - cost
    };
    let aa = a * a;
    r0 + (1.0 - r0) * aa * aa * a
}

fn beer_lambert(a: Color, d: f64) -> Color {
    Color {
        r: (-a.r * d).exp(),
        g: (-a.g * d).exp(),
        b: (-a.b * d).exp(),
    }
}

fn trace(ox: f64, oy: f64, dx: f64, dy: f64, depth: u32) -> Color {
    let mut t = 0.0;
    let sign = if scene(ox, oy).sd > 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut i = 0;
    while i < MAX_STEP && t < MAX_DISTANCE {
        let x = ox + dx * t;
        let y = oy + dy * t;
        let r = scene(x, y);
        if r.sd * sign < EPSILON {
            let mut sum = r.emissive;
            if depth < MAX_DEPTH && (r.reflectivity > 0.0 || r.eta > 0.0) {
                let mut refl = r.reflectivity;
                let (mut nx, mut ny) = gradient(x, y);
                nx *= sign;
                ny *= sign;
                if r.eta > 0.0 {
                    let eta = if sign < 0.0 {
                        r.eta
                    } else {
                        1.0 / r.eta
                    };
                    match refract(dx, dy, nx, ny, eta) {
                        Some((rx, ry)) => {
                            let cosi = -(dx * nx + dy * ny);
                            let cost = -(rx * nx + ry * ny);
                            refl = if sign < 0.0 {
                                schlick(cosi, cost, r.eta, 1.0)
                            } else {
                                schlick(cosi, cost, 1.0, r.eta)
                            };
                            sum = sum + trace(x - nx * BIAS, y - ny * BIAS, rx, ry, depth + 1) * (1.0 - refl)
                        }
                        None => {
                            refl = 1.0
                        }
                    }
                }
                if refl > 0.0 {
                    let (rx, ry) = reflect(dx, dy, nx, ny);
                    sum = sum + trace(x + nx * BIAS, y + ny * BIAS, rx, ry, depth + 1) * refl;
                }
            }
            if sign < 0.0 {
                sum = sum * beer_lambert(r.absorption, t);
            }
            return sum;
        }
        i += 1;
        t += r.sd * sign;
    }
    Color::black()
}

fn sample(rng: &mut ThreadRng, x: f64, y: f64) -> Color {
    let sum: Color = (0..N).map(|i| 2.0 * PI * (i as f64 + rng.gen_range(0.0, 1.0)) / N as f64)
        .collect::<Vec<f64>>()
        .par_iter()
        .map(|a| trace(x, y, a.cos(), a.sin(), 0))
        .sum();
    sum * (1.0 / N as f64)
}

fn main() {
    let mut img = ImageBuffer::from_pixel(W, H, Rgb([0u8, 0u8, 0u8]));
    let mut rng = rand::thread_rng();
    for x in 0..W {
        for y in 0..H {
            let xx = x as f64 / W as f64;
            let yy = y as f64 / H as f64;
            let color = sample(&mut rng, xx, yy);
            let r = min((color.r * 255.0) as u32, 255) as u8;
            let g = min((color.g * 255.0) as u32, 255) as u8;
            let b = min((color.b * 255.0) as u32, 255) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img.save("out.png").unwrap();
}
