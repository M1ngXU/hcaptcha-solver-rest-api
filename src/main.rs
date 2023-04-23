#![forbid(clippy::unwrap_used)]

use std::sync::Arc;

use rocket::http::Status;
use rocket::response::Redirect;
use rocket::serde::json::Json;
use rocket::{get, post, routes, State};
use serde::Deserialize;
use solve::solve;
use tract_onnx::prelude::TractError;
use tract_onnx::Onnx;

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
	#[error("Tract error: {0}")]
	Tract(#[from] TractError),
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
	environment: &State<Arc<Onnx>>,
) -> std::result::Result<Json<serde_json::Value>, (Status, String)> {
	let Data { prompt, images } = data.into_inner();
	match solve(environment.inner().clone(), &prompt, images).await {
		Ok(res) => Ok(Json(res)),
		Err(Error::UnknownChallenge(c)) => Err((Status::NotImplemented, c)),
		Err(Error::Tract(e)) => Err((Status::FailedDependency, format!("{e:?}"))),
		Err(Error::Networking(e)) => Err((Status::InternalServerError, format!("{e:?}"))),
		Err(Error::InternalServerError(s)) => Err((Status::InternalServerError, s)),
		Err(Error::BadRequest(s)) => Err((Status::BadRequest, s)),
	}
}

#[get("/")]
async fn get_default() -> Redirect {
	Redirect::permanent("https://hcaptcha-solver.mintlify.app/")
}

#[shuttle_runtime::main]
async fn rocket() -> shuttle_rocket::ShuttleRocket {
	let rocket = rocket::build()
		.mount("/", routes![solve_v0, get_default])
		.manage(Arc::new(tract_onnx::onnx()));

	Ok(rocket.into())
}
