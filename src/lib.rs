//! This crate provides simple API to apply [Simple Linear Image Segmentation](http://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf)
//! on all types of 3-channel images. It also support SLIC0 variation of SLIC,
//! with dynamic normalization of spectral distances.

extern crate math;
use crate::lab::{rgb_2_lab, sub, LabColor};
use image::{ImageBuffer, RgbImage, Rgba, RgbaImage};
use math::round::half_up;
use std::f64::MAX;
use std::marker::Copy;

pub mod lab;

#[derive(Copy, Clone)]
struct SuperPixel {
    label: u32,
    centroid: LabPixel,
}

struct AvgValues {
    l: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    count: Vec<f64>,
    distances: u32,
}

const X_MTX: [i64; 4] = [-1, 0, 1, 0];
const Y_MTX: [i64; 4] = [0, -1, 0, 1];
const X_MTX_8: [i64; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
const Y_MTX_8: [i64; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const RESIDUAL_ERROR_THRESHOLD: f64 = 0.0;

/// Main SLIC state struct which holds the state for each step
/// of clustering process.
///
/// This struct is created by the [`slic::get_slic`] function. See its
/// documentation for more.
///
/// [`slic::get_slic`]: ./fn.get_slic.html
pub struct Slic<'a> {
    /// result amount of superpixels (clusters).
    pub k: u32,
    /// value between 0-20. balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. In SLICO mode, this is the initial compactness.
    pub compactness: f64,
    /// run SLIC-zero, the zero compactness parameter mode of SLIC.
    pub slic_zero_mode: bool,
    /// SLIC result Vec<u32>. Labels mask produced on input image. Max label is *k*.
    pub labels: Vec<i64>,
    /// reference image target image received in [`slic::get_slic`](./fn.get_slic.html).
    pub img: &'a RgbImage,
    n: u32,
    s: f64,
    super_pixel_width: u32,
    super_pixel_height: u32,
    super_pixels: Vec<SuperPixel>,
    compactnesses: Vec<f64>,
    distances: Vec<f64>,
}

impl Slic<'_> {
    /// Mutable methods which runs SLIC computations.
    /// Applies iterations until [residual error](https://en.wikipedia.org/wiki/Errors_and_residuals) will be close to 0.
    /// Calls enforce connectivity using [Connected Components Algorithm](https://en.wikipedia.org/wiki/Connected-component_labeling)
    pub fn compute(&mut self) {
        loop {
            if self.digest() == RESIDUAL_ERROR_THRESHOLD {
                break;
            }
        }
        self.labels = self.enforce_connectivity();
    }
    /// Crates 4-channel [`image::RgbaImage`](https://docs.rs/image/0.18.0/image/type.RgbaImage.html) with `self.img` dimensions.
    /// Draws yellow borders for each cluster on transparent background.
    /// Moves result variable to caller.
    ///
    /// # Examples:
    ///
    /// ```no_run
    /// use slic::get_slic;
    /// use image::{open};
    /// fn main() {
    ///     let mut image = open("./my_image.jpg").ok().expect("Cannot open image");
    ///     let img = image
    ///         .as_mut_rgb8()
    ///         .expect("Cannot get RGB from DynamicImage");
    ///
    ///     let mut slic = get_slic(img, 30, 10.0, true);
    ///     slic.compute();
    ///     slic.get_borders_image().save("./borders.png").expect("Cannot save image on disk");
    /// }
    /// ```
    pub fn get_borders_image(&self) -> RgbaImage {
        let (width, height) = (self.img.width(), self.img.height());
        let total_size = self.n as usize;
        let (mut border_x, mut border_y, mut already_bordered) = (
            vec![0u32; total_size],
            vec![0u32; total_size],
            vec![false; total_size],
        );
        let (mut cursor, mut border_pixels_count) = (0usize, 0usize);
        let mut buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(self.img.width(), self.img.height());
        let (yellow, red) = (Rgba([255, 255, 0, 255]), Rgba([255, 0, 0, 255]));

        for j in 0..height {
            for k in 0..width {
                if self.labels[cursor] == 0 {
                    buffer.put_pixel(k, j, red);
                    cursor += 1;
                    continue;
                }

                let mut heterogeneity = 0;
                for i in 0..8 {
                    let x = (k as i64) + X_MTX_8[i];
                    let y = (j as i64) + Y_MTX_8[i];

                    if (x >= 0 && (x as u32) < width) && (y >= 0 && (y as u32) < height) {
                        let index = (y * width as i64 + x) as usize;
                        if !already_bordered[index] && self.labels[cursor] != self.labels[index] {
                            heterogeneity += 1;
                        }
                    }
                }
                if heterogeneity > 1 {
                    border_x[border_pixels_count] = k;
                    border_y[border_pixels_count] = j;
                    already_bordered[cursor] = true;
                    border_pixels_count += 1;
                }
                cursor += 1;
            }
        }

        for j in 0..border_pixels_count {
            buffer.put_pixel(border_x[j], border_y[j], yellow)
        }
        buffer
    }

    fn iter(&mut self) {
        let offset = (self.s).ceil() as u32;
        self.super_pixels.clone().into_iter().for_each(|pxl| {
            let (cx, cy, _) = pxl.centroid;
            let x1 = if cx > offset { cx - offset } else { 0 };
            let y1 = if cy > offset { cy - offset } else { 0 };
            let x2 = if cx + offset < self.img.width() {
                cx + offset
            } else {
                self.img.width() - 1
            };
            let y2 = if cy + offset < self.img.height() {
                cy + offset
            } else {
                self.img.height() - 1
            };
            let i = (pxl.label - 1) as usize;
            let compactness = if self.slic_zero_mode {
                let old_compactness = 100.0 - self.compactnesses[i];
                self.compactnesses[i] = 0.0;
                old_compactness
            } else {
                self.compactness
            };

            self.compactnesses[i] = 0.0;
            for y in y1..(y2 + 1) {
                for x in x1..(x2 + 1) {
                    let idx = (y * self.img.width() + x) as usize;
                    let prev_distance = self.distances[idx];
                    // calc distance
                    let pixel = get_lab_pixel(self.img, x, y);

                    let (color_distance, new_distance) =
                        get_distance(pxl.centroid, pixel, compactness, self.s);
                    if self.slic_zero_mode {
                        let compactness_ratio = f64::min(color_distance, 100.0);
                        if self.compactnesses[i] < compactness_ratio {
                            self.compactnesses[i] = compactness_ratio;
                        };
                    }
                    if prev_distance > new_distance {
                        self.distances[idx] = new_distance;
                        self.labels[idx] = pxl.label as i64;
                    };
                }
            }
        });
    }

    fn digest(&mut self) -> f64 {
        self.iter();
        self.recompute_centers()
    }

    fn recompute_centers(&mut self) -> f64 {
        let w = self.img.width();
        let h = self.img.height();
        let k = self.k as usize;
        let mut values = AvgValues {
            l: vec![0.0; k],
            a: vec![0.0; k],
            b: vec![0.0; k],
            x: vec![0.0; k],
            y: vec![0.0; k],
            count: vec![0.0; k],
            distances: 0,
        };
        for y in 0..h {
            for x in 0..w {
                let label_idx = (y * w + x) as usize;
                let label = self.labels[label_idx] as usize - 1;
                self.distances[label] = MAX;
                let (x, y, color) = get_lab_pixel(self.img, x, y);
                values.x[label] += x as f64;
                values.y[label] += y as f64;
                values.l[label] += color[0];
                values.a[label] += color[1];
                values.b[label] += color[2];
                values.count[label] += 1.0;
            }
        }
        self.super_pixels.iter_mut().for_each(|super_pixel| {
            let label = super_pixel.label as usize - 1;
            let (cx, cy, _) = super_pixel.centroid;
            let count = values.count[label];
            let l = half_up(values.l[label] / count, 4);
            let a = half_up(values.a[label] / count, 4);
            let b = half_up(values.b[label] / count, 4);
            let x = (values.x[label] / count).round() as u32;
            let y = (values.y[label] / count).round() as u32;
            let distance = l1_distance(cx, cy, x, y);
            values.distances += distance;
            super_pixel.centroid = (x, y, [l, a, b]);
        });
        values.distances as f64 / self.k as f64
    }

    fn enforce_connectivity(&self) -> (Vec<i64>) {
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        let image_size = w * h;
        let super_pixels_count =
            image_size / (self.super_pixel_height * self.super_pixel_width) as i64;
        let (super_pixel_size, image_size_usize) =
            (image_size / super_pixels_count, image_size as usize);
        let (mut x_coords, mut y_coords) =
            (vec![0i64; image_size_usize], vec![0i64; image_size_usize]);
        let (mut main_index, mut adjust_label, mut label) = (0usize, 1i64, 1i64);
        let mut merged_labels: Vec<i64> = vec![0; image_size_usize];
        // connected components row-by-row
        for j in 0..h {
            for k in 0..w {
                if 0 == merged_labels[main_index] {
                    merged_labels[main_index] = label;

                    x_coords[0] = k;
                    y_coords[0] = j;
                    // find adjust label
                    for n in 0usize..4 {
                        let x = x_coords[0] as i64 + X_MTX[n];
                        let y = y_coords[0] as i64 + Y_MTX[n];
                        if (x >= 0 && x < w) && (y >= 0 && y < h) {
                            let sub_index = (y * w + x) as usize;
                            if merged_labels[sub_index] > 0 {
                                adjust_label = merged_labels[sub_index]
                            }
                        }
                    }
                    // collect super_pixel, save its real pixel coords into buffer vectors
                    let (mut o, mut order) = (0, 1);
                    while o < order {
                        for n in 0..4 {
                            let x = x_coords[o] + X_MTX[n];
                            let y = y_coords[o] + Y_MTX[n];

                            if (x >= 0 && x < w) && (y >= 0 && y < h) {
                                let sub_index = (y * w + x) as usize;
                                if 0 == merged_labels[sub_index]
                                    && self.labels[main_index] == self.labels[sub_index]
                                {
                                    x_coords[order] = x;
                                    y_coords[order] = y;
                                    merged_labels[sub_index] = label;
                                    order += 1;
                                }
                            }
                        }
                        o += 1;
                    }
                    // filter super_pixels above threshold, assign them to adjust labels using buffer vectors
                    if order as i64 <= super_pixel_size >> 2 {
                        for o in 0..order {
                            let ind = (y_coords[o] * w + x_coords[o]) as usize;
                            merged_labels[ind] = adjust_label
                        }
                        label -= 1;
                    }
                    label += 1;
                }
                main_index += 1;
            }
        }
        return merged_labels;
    }
}

/// Constructor for [`slic0::Slic`] struct. Produces SLIC state object.
///
/// | Argument              | Role                                                        |
/// |:---------------------:|:-----------------------------------------------------------:|
/// | *img*                 | Image ([`image::RgbImage`]) target for SLIC.                |
/// | *num_of_super_pixels* | Approximate (+/- 3) amount of requested clusters.           |
/// | *compactness*         | balances color proximity and space proximity. Between 0-20. |
/// | *slic_zero_mode*      | run SLIC-zero, the zero compactness parameter mode of SLIC. |
///
///
/// # Examples:
///
/// ```no_run
/// use slic::get_slic;
/// use image::{open};
/// fn main() {
///     let mut image = open("./my_image.jpg").ok().expect("Cannot open image");
///     let img = image
///         .as_mut_rgb8()
///         .expect("Cannot get RGB from DynamicImage");
///
///     let mut slic = get_slic(img, 30, 10.0, true);
///     slic.compute();
///     // print labels as ascii symbols art to stdout
///     slic.labels.iter().enumerate().for_each(|(i, x)| {
///        print!("{}", ((32 + *x) as u8 as char).to_string());
///        if ((i + 1) % img.width() as usize) == 0 {
///            println!();
///        }
///     });
/// }
/// ```
///
/// [`slic0::Slic`]: ./struct.Slic.html
/// [`image::RgbImage`]: https://docs.rs/image/0.19.0/image/type.RgbImage.html
pub fn get_slic(
    img: &RgbImage,
    num_of_super_pixels: u32,
    compactness: f64,
    slic_zero_mode: bool,
) -> Slic {
    let (w, h) = img.dimensions();
    let n = w * h;
    let super_pixel_size = get_super_pixel_size(w, h, num_of_super_pixels);
    let super_pixel_edge = f64::from(super_pixel_size).sqrt().ceil() as u32;
    let mut super_pixels: Vec<SuperPixel> = Vec::with_capacity(num_of_super_pixels as usize);
    let super_pixels_per_width = (f64::from(w) / f64::from(super_pixel_edge)).round() as u32;
    let super_pixels_per_height = num_of_super_pixels / super_pixels_per_width;
    let (super_pixel_width, super_pixel_height) =
        (w / super_pixels_per_width, h / super_pixels_per_height);
    for y in 0..super_pixels_per_height {
        for x in 0..super_pixels_per_width {
            let (sx, sy) = (x * super_pixel_width, y * super_pixel_height);
            let (cx, cy) = (
                sx + (super_pixel_width as f64 * 0.5_f64).round() as u32,
                sy + (super_pixel_height as f64 * 0.5_f64).round() as u32,
            );
            let label = y * super_pixels_per_width + x + 1;
            let centroid = get_initial_centroid(img, cx, cy);
            super_pixels.push(SuperPixel { label, centroid })
        }
    }
    let k = super_pixels_per_width * super_pixels_per_height;
    Slic {
        k,
        n,
        s: (n as f64 / k as f64).sqrt(),
        super_pixel_width,
        super_pixel_height,
        compactness,
        super_pixels,
        slic_zero_mode,
        distances: vec![MAX; n as usize],
        compactnesses: vec![compactness; k as usize],
        labels: vec![-1; n as usize],
        img,
    }
}

fn get_initial_centroid(img: &RgbImage, cx: u32, cy: u32) -> LabPixel {
    let (_, pxl) = get_pixel_3x3_neighbourhood(img, cx, cy)
        .into_iter()
        .flatten()
        .fold((MAX, None), |acc, option| {
            if let Some(next_pxl) = option {
                let (prev_gradient, _) = acc;
                let (x, y, _) = next_pxl;
                let next_gradient = get_gradient_position(img, *x, *y);
                if next_gradient <= prev_gradient {
                    (next_gradient, Some(*next_pxl))
                } else {
                    acc
                }
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
            Some(get_lab_pixel(img, x, y - 1))
        } else {
            None
        },
        if y + 1 < h {
            Some(get_lab_pixel(img, x, y + 1))
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
        let (_, _, color) = *pxl;
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
        (((p1x as i64 - p2x as i64) as f64).powi(2) + ((p1y as i64 - p2y as i64) as f64).powi(2))
            .sqrt(),
        4,
    )
}

fn get_distance(p1: LabPixel, p2: LabPixel, m: f64, s: f64) -> (f64, f64) {
    let (p1x, p1y, color_p1) = p1;
    let (p2x, p2y, color_p2) = p2;
    let color_distance = get_color_distance(color_p1, color_p2);
    (
        color_distance,
        half_up(
            color_distance + m / s * get_spacial_distance((p1x, p1y), (p2x, p2y)),
            4,
        ),
    )
}

type LabPixel = (u32, u32, LabColor);

#[cfg(test)]
mod tests;
