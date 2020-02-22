use super::*;
use image::{open, ImageBuffer, Rgb, RgbImage};
use std::collections::hash_map::RandomState;
use std::collections::HashSet;

fn check_superpixels_count(slic: &mut Slic) -> bool {
    slic.k as usize
        == slic
            .labels
            .iter()
            .fold(HashSet::<i64, RandomState>::new(), |mut count, x| {
                count.insert(*x);
                count
            })
            .len()
}

fn check_slic_result(slic: &mut Slic, img: &RgbImage) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let image_size = w * h;
    let super_pixels_count = slic.k as i64;
    let (super_pixel_size, image_size_usize) =
        (image_size / super_pixels_count, image_size as usize);
    let (mut x_coords, mut y_coords) = (vec![0i64; image_size_usize], vec![0i64; image_size_usize]);
    let (mut main_index, mut label) = (0usize, 1i64);
    let mut unique_labels = HashSet::new();
    let mut merged_labels: Vec<i64> = vec![0; image_size_usize];
    // connected components row-by-row
    for j in 0..h {
        for k in 0..w {
            if 0 == merged_labels[main_index] {
                merged_labels[main_index] = label;
                x_coords[0] = k;
                y_coords[0] = j;

                let (mut o, mut order, mut label_found) = (0, 1, false);
                while o < order {
                    for n in 0..4 {
                        let x = x_coords[o] + X_MTX[n];
                        let y = y_coords[o] + Y_MTX[n];

                        if (x >= 0 && x < w) && (y >= 0 && y < h) {
                            let sub_index = (y * w + x) as usize;
                            if 0 == merged_labels[sub_index]
                                && slic.labels[main_index] == slic.labels[sub_index]
                            {
                                x_coords[order] = x;
                                y_coords[order] = y;

                                merged_labels[sub_index] = label;
                                order += 1;
                                if o == 0 && !label_found {
                                    assert_eq!(
                                        unique_labels.get(&slic.labels[main_index]).is_none(),
                                        true,
                                        "duplicate superpixel labels should not occur"
                                    );
                                    unique_labels.insert(slic.labels[main_index]);
                                    label_found = true;
                                }
                            }
                        }
                    }
                    o += 1;
                }
                assert_eq!(
                    order as i64 > super_pixel_size >> 2,
                    true,
                    "superpixels with size < threshold should not occur"
                );
                label += 1;
            }
            main_index += 1;
        }
    }
}

#[test]
fn rgb_to_lab_works() {
    let white_rgb = [255.0, 255.0, 255.0];
    let white_lab = lab::rgb_2_lab(white_rgb);
    assert_eq!(white_lab[0].round(), 100.0);
    assert_eq!(white_lab[1].round(), 0.0);
    assert_eq!(white_lab[2].round(), 0.0);

    let black_rgb = [0.0, 0.0, 0.0];
    let black_lab = lab::rgb_2_lab(black_rgb);
    assert_eq!(black_lab[0].round(), 0.0);
    assert_eq!(black_lab[1].round(), 0.0);
    assert_eq!(black_lab[2].round(), 0.0);

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
    assert_eq!(centroid.0, 89);
    assert_eq!(centroid.1, 91);
}

#[test]
fn slic_iter_works() {
    let mut image = open("./fixture/test.jpg").ok().expect("Cannot open image");
    let img = image
        .as_mut_rgb8()
        .expect("Cannot get RGB from DynamicImage");

    let mut slic = get_slic(img, 9, 10.0, false);

    slic.iter();
    let e1 = slic.recompute_centers();
    assert!(
        check_superpixels_count(&mut slic),
        "all superpixels should be present on canvas after first iteration"
    );
    slic.iter();
    let e2 = slic.recompute_centers();
    assert!(
        e1 > e2,
        "residual error decrease it's value on each SLIC iteration"
    );
    assert!(
        check_superpixels_count(&mut slic),
        "all superpixels should be present on canvas after second iteration"
    );
}

#[test]
fn slic_compute_works() {
    let mut image = open("./fixture/test.jpg").ok().expect("Cannot open image");
    let img = image
        .as_mut_rgb8()
        .expect("Cannot get RGB from DynamicImage");

    let mut slic = get_slic(img, 150, 10.0, true);
    slic.compute();
    check_slic_result(&mut slic, img);
    let borders = slic.get_borders_image();
    borders.save("./borders.png").unwrap();
}
