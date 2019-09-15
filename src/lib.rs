extern crate math;
use crate::lab::{rgb_2_lab, sub, LabColor};
use image::{Pixel, Pixels, RgbImage};
use math::round::half_up;
use std::f64::MAX;
use std::boxed::Box;

pub mod lab;

struct SuperPixel {
    label: u32,
    centroid: LabPixel,
}

const threshold:f64 = MAX;

struct Slic<'a> {
    k: u32,
    n: u32,
    s: f64,
    super_pixel_width: u32,
    super_pixel_height: u32,
    compactness: f64,
    super_pixels: Vec<SuperPixel>,
    distances: Vec<f64>,
    labels: Vec<i64>,
    img: &'a RgbImage,
}

fn get_slic(img: &RgbImage, num_of_super_pixels: u32, compactness: f64) -> Slic {
    let (w, h) = img.dimensions();
    let n = w*h;
    let super_pixel_size = get_super_pixel_size(w, h, num_of_super_pixels);
    let super_pixel_edge = f64::from(super_pixel_size).sqrt().ceil() as u32;
    let mut super_pixels: Vec<SuperPixel> = Vec::with_capacity(num_of_super_pixels as usize);
    let super_pixels_per_width = (f64::from(w) / f64::from(super_pixel_edge)).round() as u32;
    let super_pixels_per_height = num_of_super_pixels / super_pixels_per_width;
    let (super_pixel_width, super_pixel_height) = (w / super_pixels_per_width, h / super_pixels_per_height);
    for y in 0..super_pixels_per_height {
        for x in 0..super_pixels_per_width {
            let (sx, sy) = (x * super_pixel_width, y * super_pixel_height);
            let (cx, cy) = (sx + (f64::from(super_pixel_width) * 0.5_f64).round() as u32,
                            sy + (f64::from(super_pixel_height) * 0.5_f64).round() as u32);
            let label = y * x + x + 1;
            let centroid = get_initial_centroid(img, cx, cy);
            super_pixels.push(SuperPixel{
                label,
                centroid
            })
        };
    };
    Slic{
        k: super_pixels_per_width * super_pixels_per_height,
        n,
        s: (f64::from(num_of_super_pixels) / f64::from(n)).sqrt(),
        super_pixel_width,
        super_pixel_height,
        compactness,
        super_pixels,
        distances: vec![MAX; n as usize],
        labels: vec![-1; n as usize],
        img,
    }
}


fn get_initial_centroid(img: &RgbImage, cx: u32, cy: u32) -> LabPixel {
    let (_, pxl) = get_pixel_3x3_neighbourhood(img, cx, cy)
        .into_iter()
        .flatten()
        .fold((0_f64, None), |acc, option| {
            if let Some(next_pxl) = option {
                let (prev_gradient, prev_pxl) = acc;
                let (x, y, color) = next_pxl;
                let next_gradient = get_gradient_position(img, *x, *y);
                if next_gradient <= prev_gradient { (next_gradient, Some(*next_pxl)) } else { acc }
            } else {
                acc
            }
        });
    pxl.unwrap()
}


fn get_super_pixel_size(w: u32, h: u32, num: u32) -> u32 {
    (f64::from(w) * f64::from(h) / f64::from(num)).ceil() as u32
}

fn get_lab_pixel(img: &RgbImage, x: u32, y: u32) -> LabPixel {
    let pxl = *img.get_pixel(x, y);
    (
        x,
        y,
        rgb_2_lab([f64::from(pxl[0]), f64::from(pxl[1]), f64::from(pxl[2])]),
    )
}

fn get_pixel_cross_neighbourhood(
    img: &RgbImage,
    x: u32,
    y: u32,
) -> (
    Option<LabPixel>,
    Option<LabPixel>,
    Option<LabPixel>,
    Option<LabPixel>,
) {
    let (w, h) = img.dimensions();
    (
        if x != 0 {
            Some(get_lab_pixel(img, x - 1, y))
        } else {
            None
        },
        if x + 1 < w {
            Some(get_lab_pixel(img, x + 1, y))
        } else {
            None
        },
        if y != 0 {
            Some(get_lab_pixel(img, y, y - 1))
        } else {
            None
        },
        if y + 1 < h {
            Some(get_lab_pixel(img, y, y + 1))
        } else {
            None
        },
    )
}

fn get_pixel_3x3_neighbourhood(img: &RgbImage, x: u32, y: u32) -> ([[Option<LabPixel>; 3]; 3]) {
    let (w, h) = img.dimensions();
    let pxl = get_lab_pixel(img, x, y);
    let (left, right, top, bottom) = get_pixel_cross_neighbourhood(img, x, y);
    let top_left = if x != 0 && y != 0 {
        Some(get_lab_pixel(img, x - 1, y - 1))
    } else {
        None
    };
    let bottom_left = if x != 0 && y + 1 < h {
        Some(get_lab_pixel(img, x - 1, y + 1))
    } else {
        None
    };
    let top_right = if x + 1 < w && y != 0 {
        Some(get_lab_pixel(img, x + 1, y - 1))
    } else {
        None
    };
    let bottom_right = if x + 1 < w && y + 1 < h {
        Some(get_lab_pixel(img, x + 1, y + 1))
    } else {
        None
    };
    [
        [top_left, top, top_right],
        [left, Some(pxl), right],
        [bottom_left, bottom, bottom_right],
    ]
}

fn get_gradient_position(img: &RgbImage, x: u32, y: u32) -> f64 {
    let pxl = get_lab_pixel(img, x, y);
    let (left, right, top, bottom) = get_pixel_cross_neighbourhood(img, x, y);
    let left_color = get_neighbour_color_vector(&pxl, left);
    let right_color = get_neighbour_color_vector(&pxl, right);
    let top_color = get_neighbour_color_vector(&pxl, top);
    let bottom_color = get_neighbour_color_vector(&pxl, bottom);
    half_up(
        l2_norm(sub(right_color, left_color)).powf(2_f64)
            + l2_norm(sub(bottom_color, top_color)).powf(2_f64),
        4,
    )
}

fn get_neighbour_color_vector(pxl: &LabPixel, neighbour: Option<LabPixel>) -> LabColor {
    if let Some(n_pxl) = neighbour {
        let (_, _, n_color) = n_pxl;
        n_color
    } else {
        // if no neighbour return original pixel
        let (lx, ly, color) = *pxl;
        color
    }
}

fn l2_norm(y: LabColor) -> f64 {
    y.iter()
        .fold(0f64, |sum, &ey| sum + (ey.abs()).powf(2.))
        .sqrt()
}

fn l1_distance(x1: u32, y1: u32, x2: u32, y2: u32) -> u32 {
    ((x1 as i64 - x2 as i64).abs() + (y1 as i64 - y2 as i64).abs()) as u32
}

fn get_color_distance(p1: LabColor, p2: LabColor) -> f64 {
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
    half_up(
        get_color_distance(color_p1, color_p2)
            + m / S * get_spacial_distance((p1x, p1y), (p2x, p2y)),
        4,
    )
}

type LabPixel = (u32, u32, LabColor);

#[cfg(test)]
mod tests;
