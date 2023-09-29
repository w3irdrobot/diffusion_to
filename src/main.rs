use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::prelude::*;
use clap::Parser;
use sha2::{Digest, Sha256};
use tokio::fs;

use diffusion_to::prelude::*;

/// CLI for requesting and downloading AI-created images via diffusion.to
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The token for the API
    #[arg(short, long)]
    api_key: String,

    /// The prompt for the image
    #[arg(short, long)]
    prompt: String,

    /// The negative prompt for the image
    #[arg(short, long)]
    negative: Option<String>,

    /// The number of steps for the generation to use
    #[arg(short, long, default_value_t = 50)]
    steps: u16,

    /// The image model to use
    #[arg(short, long, default_value_t = String::from("beauty_realism"))]
    model: String,

    /// The size of the image
    #[arg(long, default_value_t = String::from("small"))]
    size: String,

    /// The orientation of the image
    #[arg(short, long, default_value_t = String::from("square"))]
    orientation: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let client = DiffusionClient::new(args.api_key)?;

    let mut request = ImageRequest::new(args.prompt)
        .update_steps(args.steps.try_into()?)
        .update_model(args.model.try_into()?)
        .update_size(args.size.try_into()?)
        .update_orientation(args.orientation.try_into()?);
    if let Some(negative) = args.negative {
        request = request.update_negative_prompt(negative);
    }

    let token = client.request_image(request).await?;
    // wait for up to five minutes
    let image = client
        .check_and_wait(token, Some(Duration::from_secs(300)))
        .await?;

    // process and save image
    let contents = image
        .raw
        .split(",")
        .last()
        .ok_or(anyhow!("invalid raw image data"))?;
    let binary = BASE64_STANDARD.decode(contents)?;
    let hash = Sha256::digest(&binary);
    let filename = format!("{}.png", hex::encode(hash));

    fs::write(&filename, binary).await?;

    println!("image written to {}", filename);

    Ok(())
}
