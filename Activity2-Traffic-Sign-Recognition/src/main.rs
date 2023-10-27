extern crate tract_onnx;
extern crate image;

use tract_onnx::prelude::*;
use image::GenericImageView;
use std::fs;

fn main() {
    // Load the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path("../../data/Models/traffic-sign-recognition-model.onnx")
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 30, 30)))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    // Get list of files in the Tests directory
    let paths = fs::read_dir("../../data/Test").unwrap();

    for path in paths {
        let img_path = path.unwrap().path();
        if img_path.extension().unwrap() == "jpg" {
            // Load and preprocess the image
            let img = image::open(img_path).unwrap().resize(30, 30, image::imageops::FilterType::Nearest);
            let vec: Vec<f32> = img.pixels().flat_map(|(_, _, rgb)| {
                vec![rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
            }).collect();

            // Run the model
            let output = model.run(tvec!(vec)).unwrap();

            // Get the maximum value's index = class id
            let class_id = output[0]
                .to_array_view::<f32>()
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();

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


            println!("File: {:?}, Predicted: {}", img_path, labels[class_id]);
        }
    }
}
