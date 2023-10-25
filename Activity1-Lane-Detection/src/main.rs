extern crate opencv;

use opencv::{
    core,
    highgui,
    imgproc,
    videoio,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    // Initialize a VideoCapture object to read from a video file
    let mut cap = videoio::VideoCapture::from_file("data/your_video.mp4", videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        println!("Could not open video file.");
        return Ok(());
    }
    
    // Create Mats to hold the frames and intermediary results
    let mut frame = core::Mat::default();
    let mut gray = core::Mat::default();
    let mut blurred = core::Mat::default();
    let mut edges = core::Mat::default();
    
    // Vector to hold the line segments returned by the Hough transform
    let mut lines = core::Vector::<core::Vec4i>::new();

    loop {
        // Read a frame from the video
        cap.read(&mut frame)?;
        
        // Break the loop if we have reached the end of the video
        if frame.size()?.width <= 0 {
            break;
        }

        // Convert the frame to grayscale
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Apply Gaussian blur to the grayscale frame
        imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(5, 5), 1.5, 1.5, core::BORDER_DEFAULT)?;

        // Detect edges in the image
        imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;
        
        // Run Hough Line Transform to find lines in the binary image
        imgproc::hough_lines_p(&edges, &mut lines, 1.0, core::CV_PI / 180.0, 50, 50.0, 10.0)?;

        // Draw the lines on the original frame
        for line in lines.iter() {
            imgproc::line(&mut frame, core::Point::new(line[0], line[1]), core::Point::new(line[2], line[3]), core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
        }

        // Display the frame with detected lanes
        highgui::imshow("Lane Detection", &frame)?;
        
        // Wait for 10ms, exit the loop if user presses a key
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}
