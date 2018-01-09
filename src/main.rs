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
const EPSILON: f64 = 1e-6;
const MAX_DEPTH: u32 = 3;

#[derive(Clone, Copy)]
struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn black() -> Self {
        Self {
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

#[derive(Clone, Copy, Debug)]
struct Intersection {
    point: (f64, f64),
    normal: (f64, f64),
}

trait Shape {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection>;
    fn is_inside(&self, p: (f64, f64)) -> bool;
}

struct Circle {
    cx: f64,
    cy: f64,
    r: f64,
}

impl Shape for Circle {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection> {
        let a = d.0 * d.0 + d.1 * d.1;
        let ocx = p.0 - self.cx;
        let ocy = p.1 - self.cy;
        let b = 2.0 * (ocx * d.0 + ocy * d.1);
        let c = ocx * ocx + ocy * ocy - self.r * self.r;
        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            None
        } else {
            let t1 = (-b - delta.sqrt()) / (2.0 * a);
            let t2 = (-b + delta.sqrt()) / (2.0 * a);
            let t = if t1 > EPSILON {
                t1
            } else {
                t2
            };
            if t > EPSILON {
                let x = p.0 + d.0 * t;
                let y = p.1 + d.1 * t;
                let nx = x - self.cx;
                let ny = y - self.cy;
                let len = (nx * nx + ny * ny).sqrt();
                Some(Intersection {
                    point: (x, y),
                    normal: (nx / len, ny / len),
                })
            } else {
                None
            }
        }
    }

    fn is_inside(&self, p: (f64, f64)) -> bool {
        let x = p.0 - self.cx;
        let y = p.1 - self.cy;
        x * x + y * y < self.r * self.r
    }
}

struct Plane {
    px: f64,
    py: f64,
    nx: f64,
    ny: f64,
}

impl Shape for Plane {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection> {
        let a = d.0 * self.nx + d.1 * self.ny;
        if a.abs() < EPSILON {
            None
        } else {
            let b = (self.px - p.0) * self.nx + (self.py - p.1) * self.ny;
            let t = b / a;
            if t > EPSILON {
                Some(Intersection {
                    point: (p.0 + d.0 * t, p.1 + d.1 * t),
                    normal: (self.nx, self.ny),
                })
            } else {
                None
            }
        }
    }

    fn is_inside(&self, p: (f64, f64)) -> bool {
        (p.0 - self.px) * self.nx + (p.1 - self.py) * self.ny < 0.0
    }
}

struct Polygon {
    points: Vec<(f64, f64)>, // counterclockwise
}

impl Polygon {
    fn new(p: Vec<(f64, f64)>) -> Self {
        if p.len() > 1 {
            Self {
                points: p,
            }
        } else {
            panic!("Too few points!");
        }
    }

    fn rectangle(cx: f64, cy: f64, theta: f64, sx: f64, sy: f64) -> Self {
        Self::new([(sx, -sy), (-sx, -sy), (-sx, sy), (sx, sy)].iter()
            .map(|&(x, y)| (x * theta.cos() - y * theta.sin(), x * theta.sin() + y * theta.cos()))
            .map(|(x, y)| (x + cx, y + cy))
            .collect())
    }
}

impl Shape for Polygon {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection> {
        let mut res: Option<Intersection> = None;
        for i in 0..self.points.len() {
            let a = self.points[i];
            let b = if i + 1 == self.points.len() {
                self.points[0]
            } else {
                self.points[i + 1]
            };
            let ax = a.0 - p.0;
            let ay = a.1 - p.1;
            let bx = b.0 - p.0;
            let by = b.1 - p.1;
            let product1 = ax * d.1 - d.0 * ay;
            let product2 = bx * d.1 - d.0 * by;
            if product1 * product2 < 0.0 {
                let nx = a.1 - b.1;
                let ny = b.0 - a.0;
                let len = (nx * nx + ny * ny).sqrt();
                let nx = nx / len;
                let ny = ny / len;
                let c1 = d.0 * nx + d.1 * ny;
                if c1.abs() > EPSILON {
                    let c2 = (a.0 - p.0) * nx + (a.1 - p.1) * ny;
                    let t = c2 / c1;
                    if t > EPSILON {
                        let intersect = Intersection {
                            point: (p.0 + d.0 * t, p.1 + d.1 * t),
                            normal: (nx, ny),
                        };
                        res = match res {
                            Some(i) => {
                                if distance(p, intersect.point) < distance(p, i.point) {
                                    Some(intersect)
                                } else {
                                    Some(i)
                                }
                            }
                            None => Some(intersect),
                        }
                    }
                }
            }
        }
        res
    }

    fn is_inside(&self, p: (f64, f64)) -> bool {
        for i in 0..self.points.len() {
            let a = self.points[i];
            let b = if i + 1 == self.points.len() {
                self.points[0]
            } else {
                self.points[i + 1]
            };
            let ax = b.0 - a.0;
            let ay = b.1 - a.1;
            let bx = p.0 - a.0;
            let by = p.1 - a.1;
            if ax * by - bx * ay >= 0.0 {
                return false;
            }
        }
        true
    }
}

struct UnionShape {
    a: Box<Shape + Sync>,
    b: Box<Shape + Sync>,
}

impl Shape for UnionShape {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection> {
        match (self.a.intersect(p, d), self.b.intersect(p, d)) {
            (Some(i1), Some(i2)) => {
                let d1 = distance(p, i1.point);
                let d2 = distance(p, i2.point);
                if d1 < d2 {
                    Some(i1)
                } else {
                    Some(i2)
                }
            }
            (None, r2) => r2,
            (r1, None) => r1,
        }
    }

    fn is_inside(&self, p: (f64, f64)) -> bool {
        self.a.is_inside(p) || self.b.is_inside(p)
    }
}

impl UnionShape {
    fn new(a: Box<Shape + Sync>, b: Box<Shape + Sync>) -> UnionShape {
        UnionShape {
            a: a,
            b: b,
        }
    }
}

struct IntersectShape {
    a: Box<Shape + Sync>,
    b: Box<Shape + Sync>,
}

impl Shape for IntersectShape {
    fn intersect(&self, p: (f64, f64), d: (f64, f64)) -> Option<Intersection> {
        match (self.a.intersect(p, d), self.b.intersect(p, d)) {
            (Some(i1), Some(i2)) => {
                if self.a.is_inside(i2.point) && self.b.is_inside(i1.point) {
                    let d1 = distance(p, i1.point);
                    let d2 = distance(p, i2.point);
                    if d1 < d2 {
                        Some(i1)
                    } else {
                        Some(i2)
                    }
                } else if self.a.is_inside(i2.point) {
                    Some(i2)
                } else if self.b.is_inside(i1.point) {
                    Some(i1)
                } else {
                    None
                }
            }
            (None, Some(i2)) => {
                if self.a.is_inside(i2.point) {
                    Some(i2)
                } else {
                    None
                }
            }
            (Some(i1), None) => {
                if self.b.is_inside(i1.point) {
                    Some(i1)
                } else {
                    None
                }
            }
            (None, None) => None,
        }
    }

    fn is_inside(&self, p: (f64, f64)) -> bool {
        self.a.is_inside(p) && self.b.is_inside(p)
    }
}

impl IntersectShape {
    fn new(a: Box<Shape + Sync>, b: Box<Shape + Sync>) -> IntersectShape {
        IntersectShape {
            a: a,
            b: b,
        }
    }
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
        let sign = if r.normal.0 * dx + r.normal.1 * dy < 0.0 {
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
                        sum = sum + trace(scene, x, y, rx, ry, depth + 1) * (1.0 - refl)
                    }
                    None => {
                        refl = 1.0
                    }
                }
            }
            if refl > 0.0 {
                let (rx, ry) = reflect(dx, dy, nx, ny);
                sum = sum + trace(scene, x, y, rx, ry, depth + 1) * refl;
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
        entities: vec![Entity {
            shape: Box::new(Circle {
                cx: -0.2,
                cy: -0.2,
                r: 0.1,
            }),
            emissive: Color {
                r: 10.0,
                g: 10.0,
                b: 10.0,
            },
            reflectivity: 0.0,
            eta: 0.0,
            absorption: Color::black(),
        },
        Entity {
            shape: Box::new(Polygon::rectangle(0.5, 0.5, 0.0, 0.3, 0.2)),
            emissive: Color::black(),
            reflectivity: 0.2,
            eta: 1.5,
            absorption: Color {
                r: 4.0,
                g: 4.0,
                b: 4.0,
            },
        }],
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
