use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use serde_json::json;
use smallvec::SmallVec;
use tracing::instrument;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, IntoTValue, SimplePlan, Tensor};
use tract_onnx::tract_hir::internal::InferenceOp;
use tract_onnx::Onnx;

use crate::clean_prompt::clean_prompt;
use crate::get_data::{get_image, get_model};
use crate::{Error, Result, IMAGE_DIM};

type Model = Arc<
	SimplePlan<InferenceFact, Box<dyn InferenceOp>, Graph<InferenceFact, Box<dyn InferenceOp>>>,
>;

#[instrument(skip_all)]
fn inference(model: Model, image: Vec<f32>) -> Result<bool> {
	let input = Tensor::from_shape(&[1, 3, IMAGE_DIM as usize, IMAGE_DIM as usize], &image)?;
	let mut tensors = SmallVec::new();
	tensors.push(input.into_tvalue());
	let out = model.run(tensors)?;
	let output = out[0].as_slice::<f32>()?;
	Ok(output[0] > output[1])
}

#[instrument(skip(session))]
async fn check_image(session: Model, image_id: &str) -> Result<bool> {
	let image = get_image(image_id).await?;
	tokio::task::spawn_blocking(move || inference(session, image))
		.await
		.expect("Failed to run onnx")
}

pub async fn solve(
	onnx: Arc<Onnx>,
	prompt: &str,
	images: Vec<String>,
) -> Result<serde_json::Value> {
	let label = clean_prompt(prompt)?;
	let model = get_model(label).await?;

	let model: Model = Arc::new(
		onnx.model_for_read(&mut Cursor::new(model))?
			.into_runnable()?,
	);
	let mut errors = HashMap::new();
	let mut trues = Vec::new();

	let mut handles = Vec::with_capacity(images.len());
	for image in images {
		let session = model.clone();
		handles.push((
			image.clone(),
			tokio::spawn(async move { check_image(session, &image).await }),
		));
	}
	for (id, handle) in handles {
		match handle
			.await
			.map_err(|e| Error::InternalServerError(format!("Task failed unexpectedly: {e}")))?
		{
			Ok(true) => trues.push(id),
			Err(e) => {
				errors.insert(id, e.to_string());
			}
			_ => {}
		};
	}

	Ok(json! ({"trues": trues, "errors": errors}))
}
