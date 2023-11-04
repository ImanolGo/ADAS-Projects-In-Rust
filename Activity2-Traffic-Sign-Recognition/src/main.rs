extern crate tract_onnx;
extern crate image;

use tract_onnx::prelude::*;
use image::GenericImageView;
use std::fs;
use std::path::PathBuf;

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

            // Crop to a square as in Python code
            let (width, height) = img.dimensions();
            let min_dimension = std::cmp::min(width, height);
            let cropped_img = img.crop_imm(0, 0, min_dimension, min_dimension);

            // Resize the image to 30x30
            let resized_img = img.resize_exact(30, 30, image::imageops::FilterType::Nearest);

            // Flatten and normalize pixel values
            let vec: Vec<f32> = resized_img.pixels().flat_map(|(_, _, rgb)| {
                vec![rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0]
            }).collect();

            // Confirming the size of the vector
            println!("Size of vec after resizing: {}", vec.len());

            // Reshape the vector into an ndarray with shape (1, 30, 30, 3) to match Python's (batch, height, width, channels)
            let array: ndarray::Array4<f32> = ndarray::Array::from_shape_vec((1, 30, 30, 3), vec).unwrap();

            // No need to permute axes, since we're now matching the Python order

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
            // Output the class id for verification
            println!("Predicted class ID: {}", class_id);

            // Map the class id to its label
            let labels = [
                "Speed limit (20km/h)",
                "Speed limit (30km/h)",
                "Speed limit (50km/h)",
                "Speed limit (60km/h)",
                "Speed limit (70km/h)",
                "Speed limit (80km/h)",
                "End of speed limit (80km/h)",
                "Speed limit (100km/h)",
                "Speed limit (120km/h)",
                "No passing",
                "No passing veh over 3.5 tons",
                "Right-of-way at intersection",
                "Priority road",
                "Yield",
                "Stop",
                "No vehicles",
                "Veh > 3.5 tons prohibited",
                "No entry",
                "General caution",
                "Dangerous curve left",
                "Dangerous curve right",
                "Double curve",
                "Bumpy road",
                "Slippery road",
                "Road narrows on the right",
                "Road work",
                "Traffic signals",
                "Pedestrians",
                "Children crossing",
                "Bicycles crossing",
                "Beware of ice/snow",
                "Wild animals crossing",
                "End speed + passing limits",
                "Turn right ahead",
                "Turn left ahead",
                "Ahead only",
                "Go straight or right",
                "Go straight or left",
                "Keep right",
                "Keep left",
                "Roundabout mandatory",
                "End of no passing",
                "End no passing veh > 3.5 tons"
            ];

            // Logging
            //println!("Number of labels: {}", labels.len());
            if class_id >= labels.len() {
                println!("Error: Class ID is out of bounds for labels array. Class ID: {}", class_id);
                continue;  // Skip this iteration to prevent panic
            } else {
                println!("Valid Class ID: {}", class_id);
            }

            println!("File: {:?}, Predicted: {}", img_path, labels[class_id]);
        }
    }
}
