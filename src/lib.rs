use futures_timer::Delay;
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_repr::*;
use std::time::{Duration, Instant};
use thiserror::Error;

const API_URL: &'static str = "https://diffusion.to/api/image";
const STATUS_URL: &'static str = "https://diffusion.to/api/image/status";

pub mod prelude {
    pub use super::{
        DiffusionClient, DiffusionError, DiffusionImage, ImageModel, ImageOrientation,
        ImageRequest, ImageSize, ImageSteps, ImageToken,
    };
}

#[derive(Error, Debug)]
pub enum DiffusionError {
    #[error("internal reqwest error")]
    ReqwestError(#[from] reqwest::Error),
    #[error(transparent)]
    InvalidHeader(#[from] header::InvalidHeaderValue),
    #[error("the image is not complete")]
    ImageStatusNotReady,
    #[error("unknown http error {0}")]
    UnknownHttpError(StatusCode),
    #[error("time expired without image finishing")]
    TimeExpired,
    #[error("invalid step amount")]
    InvalidStepAmount,
    #[error("invalid model")]
    InvalidModel,
    #[error("invalid size")]
    InvalidSize,
    #[error("invalid orientation")]
    InvalidOrientation,
}

pub type Result<T> = std::result::Result<T, DiffusionError>;

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

#[derive(Debug, Serialize_repr, Deserialize_repr, Clone)]
#[repr(u16)]
pub enum ImageSteps {
    Fifty = 50,
    OneHundred = 100,
    OneHundredFifty = 150,
    TwoHundred = 200,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ImageSize {
    Small,
    Medium,
    Large,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ImageOrientation {
    Square,
    Landscape,
    Portrait,
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
