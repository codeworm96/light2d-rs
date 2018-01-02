extern crate image;
extern crate rand;

use std::f64::consts::PI;
use std::cmp::min;
use image::{ImageBuffer, Rgb};
use rand::{Rng, ThreadRng};

const W: u32 = 512;
const H: u32 = 512;
const N: u32 = 64;
const MAX_STEP: u32 = 64;
const MAX_DISTANCE: f64 = 2.0;
const EPSILON: f64 = 1e-6;
const BIAS: f64 = 1e-4;
const MAX_DEPTH: u32 = 3;

struct Res {
    sd: f64,
    emissive: f64,
    reflectivity: f64,
    eta: f64,
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
        sd: circle_sdf(x, y, -0.2, -0.2, 0.1),
        emissive: 10.0,
        reflectivity: 0.0,
        eta: 0.0,
    };
    let b = Res {
        sd: box_sdf(x, y, 0.5, 0.5, 0.0, 0.3, 0.2),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let c = Res {
        sd: circle_sdf(x, y, 0.5, -0.5, 0.05),
        emissive: 20.0,
        reflectivity: 0.0,
        eta: 0.0,
    };
    let d = Res {
        sd: circle_sdf(x, y, 0.5, 0.2, 0.35),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let e = Res {
        sd: circle_sdf(x, y, 0.5, 0.8, 0.35),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let f = Res {
        sd: box_sdf(x, y, 0.5, 0.5, 0.0, 0.2, 0.1),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let g = Res {
        sd: circle_sdf(x, y, 0.5, 0.12, 0.35),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let h = Res {
        sd: circle_sdf(x, y, 0.5, 0.87, 0.35),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let i = Res {
        sd: circle_sdf(x, y, 0.5, 0.5, 0.2),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    let j = Res {
        sd: plane_sdf(x, y, 0.5, 0.5, 0.0, -1.0),
        emissive: 0.0,
        reflectivity: 0.2,
        eta: 1.5,
    };
    // a + b
    // c + d * e
    // c + (f - (g + h))
    c + i * j
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

fn trace(ox: f64, oy: f64, dx: f64, dy: f64, depth: u32) -> f64 {
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
                            sum += (1.0 - refl) * trace(x - nx * BIAS, y - ny * BIAS, rx, ry, depth + 1)
                        }
                        None => {
                            refl = 1.0
                        }
                    }
                }
                if refl > 0.0 {
                    let (rx, ry) = reflect(dx, dy, nx, ny);
                    sum += refl * trace(x + nx * BIAS, y + ny * BIAS, rx, ry, depth + 1);
                }
            }
            return sum;
        }
        i += 1;
        t += r.sd * sign;
    }
    0.0
}

fn sample(rng: &mut ThreadRng, x: f64, y: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        let a = 2.0 * PI * (i as f64 + rng.gen_range(0.0, 1.0)) / N as f64;
        sum += trace(x, y, a.cos(), a.sin(), 0);
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
