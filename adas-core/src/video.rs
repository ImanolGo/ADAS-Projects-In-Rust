//! Video in/out through the `ffmpeg` command-line tool, or plain image folders.
//!
//! Using ffmpeg as a subprocess keeps the build pure Rust (no OpenCV, no
//! libclang): any ffmpeg on the PATH can decode the input and encode MP4 or GIF.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use image::RgbImage;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
}

const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "ppm"];

fn ffmpeg_missing(err: std::io::Error) -> anyhow::Error {
    if err.kind() == std::io::ErrorKind::NotFound {
        anyhow!("ffmpeg/ffprobe not found: install ffmpeg (apt install ffmpeg, brew install ffmpeg) or use a folder of images")
    } else {
        err.into()
    }
}

/// Width, height and frame rate of a video file.
pub fn probe(path: &Path) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(ffmpeg_missing)?;
    if !out.status.success() {
        bail!("ffprobe could not read {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let fields: Vec<&str> = text.trim().split(',').collect();
    let [w, h, rate] = fields[..] else { bail!("unexpected ffprobe output: {text}") };
    let fps = match rate.split_once('/') {
        Some((n, d)) => n.parse::<f32>()? / d.parse::<f32>()?.max(1.0),
        None => rate.parse()?,
    };
    Ok(VideoInfo { width: w.parse()?, height: h.parse()?, fps })
}

/// Frames from a video file (via ffmpeg) or a folder of images (sorted by name).
pub enum FrameSource {
    Video { child: Child, stdout: BufReader<ChildStdout>, info: VideoInfo, done: bool },
    Images { paths: std::vec::IntoIter<PathBuf>, info: VideoInfo },
}

impl FrameSource {
    /// Open `path`. `start`/`duration` (seconds) trim a video; `fps` is used for image folders.
    pub fn open(path: &Path, start: Option<f32>, duration: Option<f32>, fps: f32) -> Result<Self> {
        if path.is_dir() {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("reading {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
                })
                .collect();
            paths.sort();
            let first = paths.first().ok_or_else(|| anyhow!("no images in {}", path.display()))?;
            let (width, height) = image::image_dimensions(first)?;
            return Ok(Self::Images { paths: paths.into_iter(), info: VideoInfo { width, height, fps } });
        }
        if !path.exists() {
            bail!("{} does not exist (run scripts/fetch-data.sh for the sample clips)", path.display());
        }
        let info = probe(path)?;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error"]);
        if let Some(s) = start {
            cmd.args(["-ss", &s.to_string()]);
        }
        cmd.arg("-i").arg(path);
        if let Some(d) = duration {
            cmd.args(["-t", &d.to_string()]);
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout(Stdio::piped()).stdin(Stdio::null());
        let mut child = cmd.spawn().map_err(ffmpeg_missing)?;
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Self::Video { child, stdout, info, done: false })
    }

    pub fn info(&self) -> VideoInfo {
        match self {
            Self::Video { info, .. } | Self::Images { info, .. } => *info,
        }
    }
}

impl Iterator for FrameSource {
    type Item = Result<RgbImage>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Images { paths, .. } => {
                let path = paths.next()?;
                Some(
                    image::open(&path)
                        .map(|i| i.to_rgb8())
                        .with_context(|| format!("reading {}", path.display())),
                )
            }
            Self::Video { stdout, info, done, child } => {
                if *done {
                    return None;
                }
                let mut buf = vec![0u8; (info.width * info.height * 3) as usize];
                match stdout.read_exact(&mut buf) {
                    Ok(()) => Some(Ok(RgbImage::from_raw(info.width, info.height, buf).expect("frame size"))),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        *done = true;
                        let _ = child.wait();
                        None
                    }
                    Err(e) => {
                        *done = true;
                        Some(Err(e.into()))
                    }
                }
            }
        }
    }
}

impl Drop for FrameSource {
    fn drop(&mut self) {
        if let Self::Video { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Writes frames to `.mp4`/`.mkv`/`.mov` (H.264), `.webm` (VP9), animated `.webp` or `.gif`, or a folder of PNGs.
pub enum FrameSink {
    Video { child: Child, stdin: Option<ChildStdin>, path: PathBuf },
    Pngs { dir: PathBuf, count: usize },
}

impl FrameSink {
    pub fn create(path: &Path, width: u32, height: u32, fps: f32) -> Result<Self> {
        let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        let Some(ext) = ext else {
            std::fs::create_dir_all(path)?;
            return Ok(Self::Pngs { dir: path.to_path_buf(), count: 0 });
        };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]).args([
            "-s",
            &format!("{width}x{height}"),
            "-r",
            &format!("{fps}"),
            "-i",
            "-",
        ]);
        match ext.as_str() {
            "gif" => {
                let gif_fps = fps.min(15.0);
                cmd.args([
                    "-vf",
                    &format!(
                        "fps={gif_fps},scale='min(720,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
                    ),
                    "-loop",
                    "0",
                ]);
            }
            "webm" => {
                cmd.args(["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34"]);
            }
            "webp" => {
                // Animated WebP: ~10x smaller than GIF for camera footage, and GitHub shows it inline.
                let webp_fps = fps.min(12.0);
                cmd.args([
                    "-vf",
                    &format!("fps={webp_fps},scale='min(640,iw)':-1:flags=lanczos"),
                    "-c:v",
                    "libwebp_anim",
                    "-q:v",
                    "55",
                    "-compression_level",
                    "6",
                    "-loop",
                    "0",
                ]);
            }
            "mp4" | "mkv" | "mov" => {
                cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart"]);
            }
            other => bail!("unsupported output .{other} (use .mp4, .webm, .webp, .gif, or a folder)"),
        }
        cmd.arg(path).stdin(Stdio::piped()).stdout(Stdio::null());
        let mut child = cmd.spawn().map_err(ffmpeg_missing)?;
        let stdin = child.stdin.take();
        Ok(Self::Video { child, stdin, path: path.to_path_buf() })
    }

    pub fn write(&mut self, frame: &RgbImage) -> Result<()> {
        match self {
            Self::Pngs { dir, count } => {
                frame.save(dir.join(format!("frame_{count:05}.png")))?;
                *count += 1;
            }
            Self::Video { stdin, .. } => {
                stdin
                    .as_mut()
                    .context("sink already finished")?
                    .write_all(frame.as_raw())
                    .context("ffmpeg stopped accepting frames")?;
            }
        }
        Ok(())
    }

    /// Flush and wait for the encoder. Call once at the end.
    pub fn finish(mut self) -> Result<()> {
        if let Self::Video { child, stdin, path } = &mut self {
            drop(stdin.take());
            let status = child.wait()?;
            if !status.success() {
                bail!("ffmpeg failed to write {}", path.display());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_folders_work_without_ffmpeg() {
        let dir = std::env::temp_dir().join(format!("adas-frames-{}", std::process::id()));
        let mut sink = FrameSink::create(&dir, 8, 6, 10.0).unwrap();
        for v in [10u8, 200] {
            sink.write(&RgbImage::from_pixel(8, 6, image::Rgb([v, v, v]))).unwrap();
        }
        sink.finish().unwrap();
        let source = FrameSource::open(&dir, None, None, 10.0).unwrap();
        assert_eq!(source.info(), VideoInfo { width: 8, height: 6, fps: 10.0 });
        let frames: Vec<RgbImage> = source.map(Result::unwrap).collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].get_pixel(0, 0)[0], 200);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_input_is_a_clear_error() {
        let err = FrameSource::open(Path::new("/no/such/clip.mp4"), None, None, 25.0).err().unwrap();
        assert!(err.to_string().contains("fetch-data"));
    }
}
