extern crate math;
use crate::lab::{rgb_2_lab, sub, LabColor};
use image::{Pixel, Pixels, RgbImage};
use math::round::half_up;
use std::boxed::Box;
use std::f64::MAX;
use std::u32::{MAX as U32_MAX};
use std::cmp::min;
use std::marker::{Copy};
use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::RandomState;

pub mod lab;
pub mod connectivity;

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

struct ConnectedPixels {
    target: ConnectedPixel,
    prev: Option<ConnectedPixel>,
    top: Option<ConnectedPixel>
}

struct ConnectedPixel {
    x: u32,
    y: u32,
    label: usize,
    label_index: usize
}

const threshold: f64 = MAX;

struct Slic<'a> {
    k: u32,
    n: u32,
    s: f64,
    super_pixels_per_width: u32,
    super_pixel_width: u32,
    super_pixels_per_height: u32,
    super_pixel_height: u32,
    compactness: f64,
    super_pixels: Vec<SuperPixel>,
    distances: Vec<f64>,
    labels: Vec<i64>,
    img: &'a RgbImage,
}

impl Slic<'_> {
    fn run(&mut self) {
        let offset = (self.s).ceil() as u32;
        self.super_pixels.clone().into_iter().for_each(|pxl| {
          let (cx, cy, center_color) = pxl.centroid;
          let x1 = if cx > offset { cx - offset } else { 0 };
          let y1 = if cy > offset { cy - offset } else { 0 };
          let x2 = if cx + offset < self.img.width() { cx + offset } else { self.img.width() - 1};
          let y2 = if cy + offset < self.img.height() { cy + offset } else { self.img.height() - 1};
          for y in y1..(y2+1) {
            for x in x1..(x2+1) {
              let idx = (y * self.img.width() + x) as usize;
              let prev_distance = self.distances[idx];
              // calc distance
              let pixel = get_lab_pixel(self.img, x, y);
              let new_distance = get_distance(pxl.centroid, pixel, self.compactness, self.s);
              if prev_distance > new_distance {
                self.distances[idx] = new_distance;
                self.labels[idx] = pxl.label as i64;
              };
            }
          }
        });
    }

  fn recompute_centers(&mut self)-> u32 {
    let w = self.img.width();
    let h = self.img.height();
    let k = self.k as usize;
    let mut values = AvgValues{
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
      let l = half_up((values.l[label] / count), 4);
      let a = half_up((values.a[label] / count), 4);
      let b = half_up((values.b[label] / count), 4);
      let x = (values.x[label] / count).round() as u32;
      let y = (values.y[label] / count).round() as u32;
      let distance = l1_distance(cx, cy, x, y);
      values.distances += distance;
      super_pixel.centroid = (x, y, [l, a, b]);
      println!("({} {}) label {} x {} y {} R {} G {} B {} distance {}", cx, cy, label + 1, x, y, l, a, b, distance);
    });
    values.distances /  self.k
  }

  fn enforce_connectivity(&self) -> (Vec<u32>) {
    let w = self.img.width() as i32;
    let h = self.img.height() as i32;
    let size = w * h;
    let target_supsz = size / (self.super_pixel_height * self.super_pixel_width) as i32;
    let supsz = size / target_supsz;

    let dx4: [i32; 4] = [-1, 0, 1, 0];
    let dy4: [i32; 4] = [0, -1, 0, 1];

    let mut xvec = vec![0i32; sz];
    let mut yvec = vec![0i32; sz];

    let (mut oindex, mut adjlabel, mut label) = (0usize, 0usize, 0u32);

    let mut nlabels:Vec<u32> = vec![U32_MAX; sz as usize];

    for j in 0..h {
      for k in 0..w {
        if nlabels[oindex] != U32_MAX {
          nlabels[oindex] = label;

          xvec[0] = k;
          yvec[0] = j;
          for n in 0usize..4 {
            let (x, y) = (xvec[0] as i32 + dx4[n], yvec[0] as i32 + dy4[n]);
            if (x >= 0 && x < w as i32) && (y >= 0 && y < h as i32) {
              let nindex = y  *w + x;
              if nlabels[nindex] >= 0 {
                adjlabel = nlabels[nindex]
              }
            }
          }

          let (mut c, mut count) = (0, 1);
          while c < count {
            for n in 0..4 {
              let x = xvec[c] + dx4[n];
              let y = yvec[c] + dy4[n];

              if (x >= 0 && x < w) && (y >= 0 && y < h) {
                let nindex = y*w + x;

                if nlabels[oindex] == U32_MAX && self.labels[oindex] ==  self.labels[nindex] {
                  xvec[count] = x;
                  yvec[count] = y;
                  nlabels[nindex] = label;
                  count+=1;
                }
              }
              c+=1;
            }
          }

          if count <= supsz as usize >> 2  {
            for i in 0..count {
              let ind = yvec[i]*width + xvec[i];
              nlabels[ind] = adjlabel
            }
            label-=1;
          }
          label+=1;
        }
      }
      oindex+=1;
    }
    return nlabels
  }


  // get_prev_top_and_center_pixel
  // 0x0
  // xx0
  // 000
  fn get_prev_top_and_center_pixel_label(&self, x: u32, y: u32) ->ConnectedPixels {
      let target = self.get_pixel_label(x, y).expect("Target pixel out of range");
      let prev = if x != 0 {self.get_pixel_label(x -1, y)} else {
          None
      };
      let top = if y != 0 { self.get_pixel_label(x, y-1) } else {
          None
      };
      ConnectedPixels{
          target,
          prev,
          top,
      }
  }

  fn get_pixel_label(&self, x: u32, y: u32) -> Option<ConnectedPixel> {
      if let Some(label_index) = self.get_label_index(x, y) {
          let label = self.labels[label_index] as usize - 1;
          Some(ConnectedPixel{
              x,
              y,
              label,
              label_index
          })
      } else {
          None
      }
  }

  fn get_label_index(&self, x: u32, y: u32)-> Option<usize> {
    if x >= self.img.width() {
      return None
    }
    if y >= self.img.height() {
      return None
    }
    Some((y * self.img.width() + x) as usize)
  }
}



fn get_slic(img: &RgbImage, num_of_super_pixels: u32, compactness: f64) -> Slic {
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
        super_pixels_per_width,
        super_pixel_width,
        super_pixels_per_height,
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
        .fold((MAX, None), |acc, option| {
            if let Some(next_pxl) = option {
                let (prev_gradient, prev_pxl) = acc;
                let (x, y, color) = next_pxl;
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
        (((p1x as i64 - p2x as i64) as f64).powi(2) + ((p1y as i64 - p2y as i64) as f64).powi(2)).sqrt(),
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
