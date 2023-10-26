// Import necessary opencv modules
extern crate opencv;

use opencv::{
    core,           // Core functionality, basic data structures
    highgui,        // GUI functionalities, displaying images and video
    imgproc,        // Image processing functions
    videoio,        // Video I/O, reading and writing video
    prelude::*,     // Commonly used traits and types for OpenCV in Rust
};

fn main() -> opencv::Result<()> {
    // Initialize a video capture from the given video file
    let mut cap = videoio::VideoCapture::from_file("../data/video.mp4", videoio::CAP_ANY)?;
    
    // Check if the video was opened successfully
    if !cap.is_opened()? {
        println!("Could not open video file.");
        return Ok(());
    }

    // Declare matrices to hold various stages of image processing
    let mut frame = core::Mat::default();      // Original frame from video
    let mut gray = core::Mat::default();       // Grayscale version of the frame
    let mut blurred = core::Mat::default();    // Blurred version for noise reduction
    let mut edges = core::Mat::default();      // Edge-detected version
    let mut masked_edges = core::Mat::default();// Edge-detected version with region of interest
    let mut lines = core::Vector::<core::Vec4i>::new();  // Detected lines in the image

    // Define vertices of the region of interest (trapezoidal shape) for lane detection
    let roi_vertices = [
        core::Point::new(0, frame.rows()),                 // Bottom-left
        core::Point::new(frame.cols() / 3, frame.rows() / 2),   // Top-left
        core::Point::new(2 * frame.cols() / 3, frame.rows() / 2), // Top-right
        core::Point::new(frame.cols(), frame.rows()),      // Bottom-right
    ];
    
    loop {
        // Read a new frame from the video
        cap.read(&mut frame)?;
        
        // If frame is empty (e.g., end of video), break the loop
        if frame.size()?.width <= 0 {
            break;
        }

        // Convert the original frame to grayscale for edge detection
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Apply Gaussian blur for noise reduction
        imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(5, 5), 1.5, 1.5, core::BORDER_DEFAULT)?;
        
        // Apply the Canny edge detection on the blurred image
        imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

        // Create a mask to extract the region of interest
        let size = edges.size()?;
        let mut mask = core::Mat::new_rows_cols_with_default(size.height, size.width, opencv::core::CV_8UC1, core::Scalar::all(0.0))?;

        // let pts = &[core::Point::new(0, size.height),
        //     core::Point::new(size.width, size.height),
        //     core::Point::new(size.width / 2, size.height / 2)];
        let pts = &roi_vertices;

        
        // Fill the region of interest in the mask with white (255)
        imgproc::fill_poly(
            &mut mask, 
            &core::Vector::<core::Vector::<core::Point_<i32>>>::from_iter([core::Vector::from_iter(pts.to_vec())].iter().cloned()),
            core::Scalar::new(255.0, 255.0, 255.0, 255.0),
            8,
            0,
            core::Point::new(0, 0)
        )?;
        
        
        
        // Apply the mask to the edge-detected image to keep only the region of interest
        core::bitwise_and(&edges, &mask, &mut masked_edges, &core::Mat::default())?;

        // Detect lines in the masked image using the Hough transform
        imgproc::hough_lines_p(&edges, &mut lines, 1.0, core::CV_PI / 180.0, 50, 50.0, 10.0)?;

        // Draw detected lines on the original frame
        for line in lines.iter() {
            imgproc::line(&mut frame, core::Point::new(line[0], line[1]), core::Point::new(line[2], line[3]), core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
        }

        // Display the frame with detected lines
        highgui::imshow("Lane Detection", &mut frame)?;

        // Wait for 10ms and exit loop if a key is pressed
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    // Return result
    Ok(())
}
