#![forbid(clippy::unwrap_used)]

use ort::OrtError;
use rocket::{
    get,
    http::Status,
    post,
    response::{content::RawHtml, Redirect},
    routes,
    serde::json::Json,
    uri,
};
use serde::Deserialize;
use solve::solve;
use tokio::io::AsyncReadExt;

mod clean_prompt;
mod get_data;
mod solve;

pub const IMAGE_DIM: u32 = 64;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Networking error: {0}")]
    Networking(#[from] reqwest::Error),
    #[error("Unknown challenge: {0}")]
    UnknownChallenge(String),
    #[error("Onnx error (ort): {0}")]
    Onnx(#[from] OrtError),
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Internal Server Error: {0}")]
    InternalServerError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Deserialize)]
struct Data {
    prompt: String,
    images: Vec<String>,
}

#[post("/v0", data = "<data>")]
async fn solve_v0(
    data: Json<Data>,
) -> std::result::Result<Json<serde_json::Value>, (Status, String)> {
    let Data { prompt, images } = data.into_inner();
    match solve(&prompt, images).await {
        Ok(res) => Ok(Json(res)),
        Err(Error::UnknownChallenge(c)) => Err((Status::NotImplemented, c)),
        Err(Error::Onnx(e)) => Err((Status::FailedDependency, format!("{e:?}"))),
        Err(Error::Networking(e)) => Err((Status::InternalServerError, format!("{e:?}"))),
        Err(Error::InternalServerError(s)) => Err((Status::InternalServerError, s)),
        Err(Error::BadRequest(s)) => Err((Status::BadRequest, s)),
    }
}

#[get("/v0")]
async fn get_v0() -> RawHtml<String> {
    if cfg!(debug_assertions) {
        let mut html = String::new();
        tokio::fs::File::open("src/index.html")
            .await
            .unwrap()
            .read_to_string(&mut html)
            .await
            .unwrap();
        let mut readme = String::new();
        tokio::fs::File::open("README.md")
            .await
            .unwrap()
            .read_to_string(&mut readme)
            .await
            .unwrap();
        RawHtml(html.replace("MARKDOWN_CONTENT", &format!("{readme:?}")))
    } else {
        RawHtml(include_str!("index.html").replace(
            "MARKDOWN_CONTENT",
            &format!("{:?}", include_str!("../README.md")),
        ))
    }
}

#[get("/")]
async fn get_default() -> Redirect {
    Redirect::temporary(uri!(get_v0))
}

#[shuttle_runtime::main]
async fn rocket() -> shuttle_rocket::ShuttleRocket {
    let rocket = rocket::build().mount("/", routes![solve_v0, get_default, get_v0]);

    Ok(rocket.into())
}
