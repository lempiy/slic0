use math::round::half_up;
use std::ops;

pub type LabColor = [f64; 3];

pub fn sub(color: LabColor, other: LabColor) -> LabColor {
    [color[0] - other[0], color[1] - other[1], color[2] - other[2]]
}

// Illuminant and reference angle for output values: D65 2°
pub fn rgb_2_lab(color: [f64; 3]) -> LabColor {
    let mut rgb = [0.0; 3];
    for i in 0..color.len() {
        rgb[i] = rgb_stab(color[i]);
    }
    let mut xyz = [0.0; 3];
    xyz[0] = xyz_stab(half_up(rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805, 4) / 95.047);
    xyz[1] = xyz_stab(half_up(rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722, 4) / 100.0);
    xyz[2] = xyz_stab(half_up(rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9504, 4) / 108.883);

    [
        half_up((116.0 * xyz[1]) - 16.0, 4),
        half_up(500.0 * (xyz[0] - xyz[1]), 4),
        half_up(200.0 * (xyz[1] - xyz[2]), 4),
    ]
}

fn xyz_stab(c: f64) -> f64 {
    if c > 0.008856 {
        c.powf(0.3333333333333333)
    } else {
        (7.787 * c) + (16.0 / 116.0)
    }
}

fn rgb_stab(c: f64) -> f64 {
    let d = c / 255.0;
    if d > 0.04045 {
        ((d + 0.055) / 1.055).powf(2.4) * 100.0
    } else {
        d / 12.92 * 100.0
    }
}
