use std::collections::HashMap;

use cached::proc_macro::{cached, once};
use image::imageops::FilterType;
use serde::Deserialize;
use tracing::instrument;

use crate::{Error, Result, IMAGE_DIM};

#[derive(Debug, Clone, Deserialize)]
struct LabelAlias {
    en: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Objects {
    label_alias: HashMap<String, LabelAlias>,
}

#[once(time = 3600, result = true)]
#[instrument]
async fn get_objects() -> Result<HashMap<String, String>> {
    serde_yaml::from_slice::<Objects>(
        &reqwest::get("https://github.com/QIN2DIM/hcaptcha-challenger/raw/main/src/objects.yaml")
            .await?
            .bytes()
            .await?,
    )
    .map(|h| {
        h.label_alias
            .into_iter()
            .flat_map(|(k, v)| v.en.into_iter().map(|v| (v, k.clone())).collect::<Vec<_>>())
            .collect()
    })
    .map_err(|e| Error::InternalServerError(format!("Serde error: {e}")))
}

#[cached(result = true)]
#[instrument]
pub async fn get_model(name: String) -> Result<Vec<u8>> {
    let res = reqwest::get(format!(
        "https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{}.onnx",
        get_objects()
            .await?
            .get(&name)
            .ok_or_else(|| Error::UnknownChallenge(name.clone()))?
    ))
    .await?;
    if res.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(Error::UnknownChallenge(name));
    }
    Ok(res.error_for_status()?.bytes().await?.to_vec())
}

#[instrument]
pub async fn get_image(id: &str) -> Result<Vec<f32>> {
    let image = reqwest::ClientBuilder::new().user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36").build()?.get(format!("https://imgs.hcaptcha.com/{id}")).send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
        let mut image = image::load_from_memory(&image)
            .map_err(|e| Error::InternalServerError(format!("Loading image error: {e}")))?;
        image = image.resize_exact(IMAGE_DIM, IMAGE_DIM, FilterType::Nearest);
        let (mut r, g, b) = image
            .into_rgb8()
            .pixels()
            .map(|p| p.0.map(|c| c as f32 / 255.0))
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut r, mut g, mut b), [cr, cg, cb]| {
                    r.push(cr);
                    g.push(cg);
                    b.push(cb);
                    (r, g, b)
                },
            );
        r.extend(g);
        r.extend(b);
        Ok(r)
    })
    .await
    .map_err(|e| {
        Error::InternalServerError(format!("Transforming image failed unexpectedly: {e}"))
    })?
}
