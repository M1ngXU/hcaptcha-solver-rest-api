use std::collections::HashMap;
use std::mem::transmute;
use std::sync::Arc;

use ndarray::{ArrayBase, IxDynImpl};
use ort::tensor::InputTensor;
use ort::{Environment, InMemorySession, OrtError, SessionBuilder};
use serde_json::json;
use tracing::instrument;

use crate::clean_prompt::clean_prompt;
use crate::get_data::{get_image, get_model};
use crate::{Error, Result, IMAGE_DIM};

#[instrument(skip_all)]
fn inference(session: Arc<InMemorySession<'static>>, image: Vec<f32>) -> Result<bool> {
	let input = InputTensor::FloatTensor(
		ArrayBase::from_shape_vec(
			IxDynImpl::from(vec![1_usize, 3, IMAGE_DIM as usize, IMAGE_DIM as usize]),
			image,
		)
		.map_err(|e| {
			Error::InternalServerError(format!(
				"Image couldn't be turned into a tensor (shape error): {e}"
			))
		})?,
	);
	let output = session.run([input])?[0].try_extract::<f32>()?;
	output
		.view()
		.to_slice()
		.map(|s| s[0] > s[1])
		.ok_or_else(|| Error::Onnx(OrtError::PointerShouldNotBeNull(output.view().to_string())))
}

#[instrument(skip(session))]
async fn check_image(session: Arc<InMemorySession<'static>>, image_id: &str) -> Result<bool> {
	let image = get_image(image_id).await?;
	tokio::task::spawn_blocking(move || inference(session, image))
		.await
		.expect("Failed to run onnx")
}

pub async fn solve(
	environment: Arc<Environment>,
	prompt: &str,
	images: Vec<String>,
) -> Result<serde_json::Value> {
	let label = clean_prompt(prompt)?;
	let model = get_model(label).await?;

	// This is safe because the model is only dropped after the session is dropped.
	// Is is also ensured that nobody has a strong reference to the session anymore.
	// This is required to use the session from multiple threads, while the only
	// reason that `InMemorySession` has a liftime parameter is because the model
	// passed is a reference to the slice, not the (owned) `Vec`.

	let session = Arc::new(unsafe {
		transmute::<_, InMemorySession<'static>>(
			SessionBuilder::new(&environment)?.with_model_from_memory(&model)?,
		)
	});
	let mut errors = HashMap::new();
	let mut trues = Vec::new();

	let mut handles = Vec::with_capacity(images.len());
	for image in images {
		let session = session.clone();
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

	// Ensure that nobody has a reference to the session.
	if Arc::try_unwrap(session).is_ok() {
		std::mem::drop(model);
	} else {
		std::mem::forget(model);
		return Err(Error::InternalServerError(
			"Somebody still has a strong reference to the session while destroying the underlying \
			 model. To prevent UB, the model will be leaked (with its memory)!"
				.to_string(),
		))?;
	}
	Ok(json! ({"trues": trues, "errors": errors}))
}
