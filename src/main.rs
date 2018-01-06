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
const BIAS: f64 = 1e-4;
const MAX_DEPTH: u32 = 3;

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct Intersection {
    point: (f64, f64),
    normal: (f64, f64),
}

trait Shape {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection>;
    fn is_inside(&self, p: (f64, f64)) -> bool;
}

struct EntityIntersection {
    point: (f64, f64),
    normal: (f64, f64),
    emissive: Color,
    reflectivity: f64,
    eta: f64,
    absorption: Color,
}

struct Entity {
    shape: Box<Shape + Sync>,
    emissive: Color,
    reflectivity: f64,
    eta: f64,
    absorption: Color,
}

impl Entity {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<EntityIntersection> {
        self.shape.intersect(p, d).map(|intersection| EntityIntersection {
            point: intersection.point,
            normal: intersection.normal,
            emissive: self.emissive.clone(),
            reflectivity: self.reflectivity,
            eta: self.eta,
            absorption: self.absorption,
        })
    }
}

struct Scene {
    entities: Vec<Entity>,
}

fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    (dx * dx + dy * dy).sqrt()
}

impl Scene {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<EntityIntersection> {
        let mut res: Option<EntityIntersection> = None;
        for e in &self.entities {
            if let Some(intersection) = e.intersect(p, d) {
                res = match res {
                    Some(r) => {
                        if distance(p, r.point) > distance(p, intersection.point) {
                            Some(intersection)
                        } else {
                            Some(r)
                        }
                    }
                    None => Some(intersection),
                }
            }
        }
        res
    }
}

fn reflect(ix: f64, iy: f64, nx: f64, ny: f64) -> (f64, f64) {
    let dot2 = (ix * nx + iy * ny) * 2.0;
    (ix - dot2 * nx, iy - dot2 * ny)
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

fn trace(scene: &Scene, ox: f64, oy: f64, dx: f64, dy: f64, depth: u32) -> Color {
    if let Some(r) = scene.intersect((ox, oy), (dx, dy)) {
        let sign = if r.normal.0 * dx + r.normal.1 * dy > 0.0 {
            1.0
        } else {
            -1.0
        };
        let mut sum = r.emissive;
        if depth < MAX_DEPTH && (r.reflectivity > 0.0 || r.eta > 0.0) {
            let mut refl = r.reflectivity;
            let (x, y) = r.point;
            let nx = r.normal.0 * sign;
            let ny = r.normal.1 * sign;
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
                        sum = sum + trace(scene, x - nx * BIAS, y - ny * BIAS, rx, ry, depth + 1) * (1.0 - refl)
                    }
                    None => {
                        refl = 1.0
                    }
                }
            }
            if refl > 0.0 {
                let (rx, ry) = reflect(dx, dy, nx, ny);
                sum = sum + trace(scene, x + nx * BIAS, y + ny * BIAS, rx, ry, depth + 1) * refl;
            }
        }
        if sign < 0.0 {
            sum = sum * beer_lambert(r.absorption, distance((ox, oy), r.point));
        }
        sum
    } else {
        Color::black()
    }
}

fn sample(scene: &Scene, rng: &mut ThreadRng, x: f64, y: f64) -> Color {
    let sum: Color = (0..N).map(|i| 2.0 * PI * (i as f64 + rng.gen_range(0.0, 1.0)) / N as f64)
        .collect::<Vec<f64>>()
        .par_iter()
        .map(|a| trace(scene, x, y, a.cos(), a.sin(), 0))
        .sum();
    sum * (1.0 / N as f64)
}

fn main() {
    let mut img = ImageBuffer::from_pixel(W, H, Rgb([0u8, 0u8, 0u8]));
    let mut rng = rand::thread_rng();
    let scene = Scene {
        entities: vec![],
    };
    for x in 0..W {
        for y in 0..H {
            let xx = x as f64 / W as f64;
            let yy = y as f64 / H as f64;
            let color = sample(&scene, &mut rng, xx, yy);
            let r = min((color.r * 255.0) as u32, 255) as u8;
            let g = min((color.g * 255.0) as u32, 255) as u8;
            let b = min((color.b * 255.0) as u32, 255) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img.save("out.png").unwrap();
}
