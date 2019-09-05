extern crate math;
use image::{Pixel, Pixels, RgbImage};
use math::round::half_up;
use crate::lab::rgb_2_lab;

pub mod lab;

fn get_lab_pixel(img: &RgbImage, x: u32, y: u32) -> LabPixel {
    let pxl = *img.get_pixel(x, y);
    (x, y, rgb_2_lab([f64::from(pxl[0]), f64::from(pxl[1]), f64::from(pxl[2])]))
}

fn get_gradient_position(img: &RgbImage, x: u32, y: u32) -> f64 {
    let (lx, ly, color) = get_lab_pixel(img, x, y);
    let g = 
}

fn l2_norm(y: [f64; 3]) -> f64 {
    y.iter()
        .fold(0f64, |sum, &ey| sum + (ey.abs()).powf(2.))
        .sqrt()
}

fn get_color_distance(p1: [f64; 3], p2: [f64; 3]) -> f64 {
    half_up(
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt(),
        4,
    )
}

fn get_spacial_distance(p1: (u32, u32), p2: (u32, u32)) -> f64 {
    let (p1x, p1y) = p1;
    let (p2x, p2y) = p2;
    half_up(
        (((p1x - p2x) as f64).powi(2) + ((p1y - p2y) as f64).powi(2)).sqrt(),
        4,
    )
}

fn get_distance(p1: LabPixel, p2: LabPixel, m: f64, S: f64) -> f64 {
    let (p1x, p1y, color_p1) = p1;
    let (p2x, p2y, color_p2) = p2;
    get_color_distance(color_p1, color_p2) + m / S * get_spacial_distance((p1x, p1y), (p2x, p2y))
}

type LabPixel = (u32, u32, [f64; 3]);

#[cfg(test)]
mod tests;
