//! # diffusion_to
//!
//! `diffusion_to` is a library for interacting with the API for [diffusion.to](https://diffusion.to).
//! This API makes it easy to create AI-generated images and download them in a base64 format to be used
//! however is needed. All options available in the UI are available through the library, using enums where
//! possible to prevent invalid requests from being made.
//!
//! ## Example
//!
//! Basic usage:
//!
//! ```
//! let client = DiffusionClient::new(args.api_key)?;
//!
//! let request = ImageRequest::new(args.prompt)
//!     .update_steps(args.steps.try_into()?)
//!     .update_model(args.model.try_into()?)
//! let token = client.request_image(request).await?;
//!
//! // wait for up to five minutes
//! let image = client
//!     .check_and_wait(token, Some(Duration::from_secs(300)))
//!     .await?;
//!
//! println!("{}", iamge.raw)
//! ```

use futures_timer::Delay;
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_repr::*;
use std::{
    fmt::Display,
    time::{Duration, Instant},
};
use thiserror::Error;

const API_URL: &'static str = "https://diffusion.to/api/image";
const STATUS_URL: &'static str = "https://diffusion.to/api/image/status";

pub mod prelude {
    pub use super::{
        DiffusionClient, DiffusionError, DiffusionImage, ImageModel, ImageOrientation,
        ImageRequest, ImageSize, ImageSteps, ImageToken,
    };
}

/// Potential errors returned from the library
#[derive(Error, Debug)]
pub enum DiffusionError {
    /// Errors returned from the underlying reqwest library
    #[error("internal reqwest error")]
    ReqwestError(#[from] reqwest::Error),
    /// An invalid header
    #[error(transparent)]
    InvalidHeader(#[from] header::InvalidHeaderValue),
    /// Image has not been fully created yet
    #[error("the image is not complete")]
    ImageStatusNotReady,
    /// Unknown HTTP error returned from the API
    #[error("unknown http error {0}")]
    UnknownHttpError(StatusCode),
    /// The image was not created within the timeout
    #[error("time expired without image finishing")]
    TimeExpired,
    /// Invalid step amount given
    #[error("invalid step amount")]
    InvalidStepAmount,
    /// Invalid model given
    #[error("invalid model")]
    InvalidModel,
    /// Invalid size given
    #[error("invalid size")]
    InvalidSize,
    /// Invalid orientation given
    #[error("invalid orientation")]
    InvalidOrientation,
}

pub type Result<T> = std::result::Result<T, DiffusionError>;

/// The client used to interact with the diffusion.to API
pub struct DiffusionClient {
    api: Client,
}

impl DiffusionClient {
    pub fn new(key: String) -> Result<Self> {
        let mut headers = header::HeaderMap::new();

        let bearer = format!("Bearer {}", key);
        let mut key = header::HeaderValue::from_str(&bearer)?;
        key.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, key);

        headers.insert(header::ACCEPT, "application/json".try_into()?);

        let api = Client::builder().default_headers(headers).build()?;

        Ok(Self { api })
    }

    /// Request an image be created, using the given request to fill out the parameters
    /// for the API image to create. It returns a token that can then be used to check
    /// the status of the image and received the image when complete.
    pub async fn request_image(&self, request: ImageRequest) -> Result<ImageToken> {
        let body = self
            .api
            .post(API_URL)
            .json(&request)
            .send()
            .await?
            .json::<TokenBody>()
            .await?;

        Ok(body.into())
    }

    /// Check the status of the image using the token received from
    /// a [`request_image()`](DiffusionClient::request_image) call
    pub async fn check_status(&self, token: ImageToken) -> Result<DiffusionImage> {
        let res = self
            .api
            .post(STATUS_URL)
            .json(&TokenBody::from(token))
            .send()
            .await?;

        match res.status() {
            StatusCode::NO_CONTENT => Err(DiffusionError::ImageStatusNotReady),
            StatusCode::CREATED => Ok(res.json::<StatusResponse>().await?.data),
            code => Err(DiffusionError::UnknownHttpError(code)),
        }
    }

    /// Check the status of the image and wait for a maximum amount of time for the image
    /// to complete before returning the image response. This method will continue to poll
    /// every five seconds until either the image has been completed or the max time is hit.
    /// If `None` is passed for maximum time, then the method will poll indefinitely until the
    /// image is complete.
    pub async fn check_and_wait(
        &self,
        token: ImageToken,
        max_wait_time: Option<Duration>,
    ) -> Result<DiffusionImage> {
        let time_threshold = max_wait_time.map(|d| Instant::now() + d);
        loop {
            match self.check_status(token.clone()).await {
                Ok(image) => return Ok(image),
                // utxo-suggested poll duration is five seconds
                _ => match time_threshold {
                    Some(t) if Instant::now() >= t => return Err(DiffusionError::TimeExpired),
                    _ => Delay::new(Duration::from_secs(5)).await,
                },
            }
        }
    }
}

/// An image request to notify the API of the parameters of
/// the image to create
#[derive(Debug, Serialize)]
pub struct ImageRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative: Option<String>,
    steps: ImageSteps,
    model: ImageModel,
    size: ImageSize,
    orientation: ImageOrientation,
}

impl ImageRequest {
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            negative: None,
            steps: ImageSteps::Fifty,
            model: ImageModel::BeautyRealism,
            size: ImageSize::Small,
            orientation: ImageOrientation::Landscape,
        }
    }

    pub fn update_negative_prompt(mut self, prompt: String) -> Self {
        self.negative = Some(prompt);
        self
    }

    pub fn update_steps(mut self, steps: ImageSteps) -> Self {
        self.steps = steps;
        self
    }

    pub fn update_model(mut self, model: ImageModel) -> Self {
        self.model = model;
        self
    }

    pub fn update_size(mut self, size: ImageSize) -> Self {
        self.size = size;
        self
    }

    pub fn update_orientation(mut self, orientation: ImageOrientation) -> Self {
        self.orientation = orientation;
        self
    }
}

/// The available steps provided through the API
#[derive(Debug, Serialize_repr, Deserialize_repr, Clone)]
#[repr(u16)]
pub enum ImageSteps {
    Fifty = 50,
    OneHundred = 100,
    OneHundredFifty = 150,
    TwoHundred = 200,
}

impl Display for ImageSteps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fifty => write!(f, "50"),
            Self::OneHundred => write!(f, "100"),
            Self::OneHundredFifty => write!(f, "150"),
            Self::TwoHundred => write!(f, "200"),
        }
    }
}

impl TryFrom<u16> for ImageSteps {
    type Error = DiffusionError;

    fn try_from(value: u16) -> std::result::Result<Self, Self::Error> {
        match value {
            50 => Ok(Self::Fifty),
            100 => Ok(Self::OneHundred),
            150 => Ok(Self::OneHundredFifty),
            200 => Ok(Self::TwoHundred),
            _ => Err(DiffusionError::InvalidStepAmount),
        }
    }
}

#[cfg(feature = "clap")]
impl clap::ValueEnum for ImageSteps {
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        use clap::builder::PossibleValue;

        match self {
            Self::Fifty => Some(PossibleValue::new("50")),
            Self::OneHundred => Some(PossibleValue::new("100")),
            Self::OneHundredFifty => Some(PossibleValue::new("150")),
            Self::TwoHundred => Some(PossibleValue::new("200")),
        }
    }

    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Fifty,
            Self::OneHundred,
            Self::OneHundredFifty,
            Self::TwoHundred,
        ]
    }
}

/// The available image models provided through the API
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ImageModel {
    BeautyRealism,
    AestheticRealism,
    AnimeRealism,
    AnalogRealism,
    DreamReality,
    StableDiffusion,
    ToonAnimated,
    FantasyAnimated,
}

impl Display for ImageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeautyRealism => write!(f, "beauty_realism"),
            Self::AestheticRealism => write!(f, "aesthetic_realism"),
            Self::AnimeRealism => write!(f, "anime_realism"),
            Self::AnalogRealism => write!(f, "analog_realism"),
            Self::DreamReality => write!(f, "dream_reality"),
            Self::StableDiffusion => write!(f, "stable_diffusion"),
            Self::ToonAnimated => write!(f, "toon_animated"),
            Self::FantasyAnimated => write!(f, "fantasy_animated"),
        }
    }
}

impl TryFrom<String> for ImageModel {
    type Error = DiffusionError;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        match value.as_str() {
            "beauty_realism" => Ok(Self::BeautyRealism),
            "aesthetic_realism" => Ok(Self::AestheticRealism),
            "anime_realism" => Ok(Self::AnimeRealism),
            "analog_realism" => Ok(Self::AnalogRealism),
            "dream_reality" => Ok(Self::DreamReality),
            "stable_diffusion" => Ok(Self::StableDiffusion),
            "toon_animated" => Ok(Self::ToonAnimated),
            "fantasy_animated" => Ok(Self::FantasyAnimated),
            _ => Err(DiffusionError::InvalidModel),
        }
    }
}

/// The available image sizes provided through the API
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ImageSize {
    Small,
    Medium,
    Large,
}

impl Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
        }
    }
}

impl TryFrom<String> for ImageSize {
    type Error = DiffusionError;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        match value.as_str() {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err(DiffusionError::InvalidSize),
        }
    }
}

/// The available iamge orientations provided through the API
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ImageOrientation {
    Square,
    Landscape,
    Portrait,
}

impl Display for ImageOrientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Square => write!(f, "square"),
            Self::Landscape => write!(f, "landscape"),
            Self::Portrait => write!(f, "portrait"),
        }
    }
}

impl TryFrom<String> for ImageOrientation {
    type Error = DiffusionError;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        match value.as_str() {
            "square" => Ok(Self::Square),
            "landscape" => Ok(Self::Landscape),
            "portrait" => Ok(Self::Portrait),
            _ => Err(DiffusionError::InvalidOrientation),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TokenBody {
    token: String,
}

impl From<ImageToken> for TokenBody {
    fn from(value: ImageToken) -> Self {
        Self { token: value.0 }
    }
}

/// A token returned from the API that is used to check
/// the status of the image and get the image when completed
#[derive(Clone)]
pub struct ImageToken(String);

impl From<TokenBody> for ImageToken {
    fn from(value: TokenBody) -> Self {
        Self(value.token)
    }
}

#[derive(Deserialize, Clone)]
struct StatusResponse {
    data: DiffusionImage,
}

/// The image response returned from the API when the
/// image is complete
#[derive(Deserialize, Debug, Clone)]
pub struct DiffusionImage {
    pub id: u64,
    pub steps: ImageSteps,
    pub size: ImageSize,
    pub model: ImageModel,
    pub credits_used: u64,
    pub created_at: String,
    pub updated_at: String,
    pub raw: String,
}
