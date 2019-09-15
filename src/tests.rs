use super::*;
use image::{ImageBuffer, Rgb, RgbImage};

#[test]
fn rgb_to_lab_works() {
    let white_rgb = [255.0, 255.0, 255.0];
    let white_lab = lab::rgb_2_lab(white_rgb);
    assert_eq!(white_lab[0].round(), 100.0);
    assert_eq!(white_lab[1].round(), 0.0);
    assert_eq!(white_lab[2].round(), 0.0);

    let black_rgb = [0.0, 0.0, 0.0];
    let black_lab = lab::rgb_2_lab(black_rgb);
    assert_eq!(black_rgb[0].round(), 0.0);
    assert_eq!(black_rgb[1].round(), 0.0);
    assert_eq!(black_rgb[2].round(), 0.0);

    let orange_rgb = [255.0, 122.0, 0.0];
    let orange_lab = lab::rgb_2_lab(orange_rgb);

    assert_eq!(orange_lab[0].round(), 66.0);
    assert_eq!(orange_lab[1].round(), 46.0);
    assert_eq!(orange_lab[2].round(), 73.0);
}

#[test]
fn get_lab_pixel_works() {
    let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(100, 100);
    image.put_pixel(50, 50, Rgb([255, 255, 255]));
    let (x1, y1, color1) = get_lab_pixel(&image, 50, 50);
    assert_eq!(x1, 50);
    assert_eq!(y1, 50);
    assert_eq!(color1[0].round(), 100.0);
    assert_eq!(color1[1].round(), 0.0);
    assert_eq!(color1[2].round(), 0.0);
}

#[test]
fn get_initial_centroid_works() {
    let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(1000, 500);
    image.put_pixel(8, 8, Rgb([255, 32, 255]));
    image.put_pixel(20, 20, Rgb([31, 32, 255]));
    image.put_pixel(90, 89, Rgb([0, 32, 255]));
    image.put_pixel(90, 90, Rgb([31, 32, 255]));
    image.put_pixel(91, 91, Rgb([0, 255, 0]));
    image.put_pixel(92, 91, Rgb([255, 255, 0]));
    let centroid = get_initial_centroid(&image, 90, 90);
    println!("{} {} color: {} {} {}", centroid.0, centroid.1, centroid.2[0], centroid.2[1], centroid.2[2]);
    assert_eq!(centroid.0, 89);
    assert_eq!(centroid.1, 91);
}
