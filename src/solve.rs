use std::{collections::HashMap, sync::Arc};

use ndarray::{ArrayBase, IxDynImpl};
use ort::{
    tensor::{InputTensor, OrtOwnedTensor},
    Environment, OrtError, SessionBuilder,
};
use serde_json::json;
use tracing::instrument;

use crate::{
    clean_prompt::clean_prompt,
    get_data::{get_image, get_model},
    Error, Result, IMAGE_DIM,
};

#[instrument]
async fn check_image(model: &[u8], image_id: &str) -> Result<bool> {
    let environment = Arc::new(Environment::builder().build()?);
    let session = SessionBuilder::new(&environment)?.with_model_from_memory(model)?;
    let image = get_image(image_id).await?;
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
    let output: OrtOwnedTensor<f32, _> = session.run([input])?[0].try_extract()?;
    output
        .view()
        .to_slice()
        .map(|s| s[0] > s[1])
        .ok_or_else(|| Error::Onnx(OrtError::PointerShouldBeNull(output.view().to_string())))
}

pub async fn solve(prompt: &str, images: Vec<String>) -> Result<serde_json::Value> {
    let label = clean_prompt(prompt)?;
    let model = Arc::new(get_model(label).await?);
    let mut errors = HashMap::new();
    let mut trues = Vec::new();

    let mut handles = Vec::with_capacity(images.len());
    for image in images {
        let model = model.clone();
        handles.push((
            image.clone(),
            tokio::spawn(async move { check_image(&model, &image).await }),
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
