extern crate tract_onnx;
extern crate image;

use tract_onnx::prelude::*;
use image::GenericImageView;
use std::fs;
use std::path::PathBuf;

fn get_speed_limit_from_sign(class_id: usize) -> Option<u32> {
    // A mock-up dictionary of class ids to speed limits
    let speed_limit_signs: std::collections::HashMap<usize, u32> = vec![
        // (class_id, speed_limit)
        (0, 20),
        (1, 30),
        (2, 50),
        (3, 60),
        (4, 70),
        (5, 80),
        (7, 100),
        (8, 120),
        // ... add as many as you have in your model
    ].into_iter().collect();

    // Return the speed limit if the class id is for a speed limit sign
    speed_limit_signs.get(&class_id).copied()
}

fn main() {
    // Load the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path("../data/Models/traffic-sign-recognition-model.onnx").unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();

     // Get list of file paths in the directory
     let mut paths: Vec<PathBuf> = fs::read_dir("../data/Test")
     .unwrap()
     .map(|res| res.map(|e| e.path()))
     .collect::<Result<Vec<_>, std::io::Error>>()
     .unwrap();

    // Sort the paths
    paths.sort();


    // Iterate over the sorted paths
    for img_path in paths {
        if img_path.extension().map(|ext| ext == "png").unwrap_or(false) {
            // Load the image
            let img = image::open(img_path.clone()).unwrap();

            // Resize the image to 30x30
            let resized_img = img.resize_exact(30, 30, image::imageops::FilterType::Nearest);

            // Flatten and normalize pixel values
            let vec: Vec<f32> = resized_img.pixels().flat_map(|(_, _, rgb)| {
                vec![rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0]
            }).collect();


            // Reshape the vector into an ndarray with shape (1, 30, 30, 3) to match Python's (batch, height, width, channels)
            let array: ndarray::Array4<f32> = ndarray::Array::from_shape_vec((1, 30, 30, 3), vec).unwrap();

            // Convert the array into a tensor
            let tensor = array.into_dyn().into_tensor();

            // Run the model - the tensor needs to be converted into a TValue
            let output = model.run(tvec!(tensor.into())).unwrap();


            // Process the output as in Python to find the predicted class ID
            let class_id = output[0]
                .to_array_view::<f32>()
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();


            println!("File: {:?}, Class ID: {}", img_path, class_id);

            if let Some(speed_limit) = get_speed_limit_from_sign(class_id) {
                println!("The current speed limit is: {} km/h", speed_limit);
            } else {
                println!("The detected sign is not a speed limit sign.");
            }
        }
    }
}
