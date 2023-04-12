#![feature(is_some_and)]

#[macro_use]
extern crate anyhow;
#[macro_use]
extern crate rocket;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate tracing;

use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use cached::proc_macro::{cached, once};
use image::imageops::FilterType;
use ndarray::{ArrayBase, IxDynImpl};
use ort::{
    tensor::{InputTensor, OrtOwnedTensor},
    Environment, SessionBuilder,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use rocket::serde::json::Json;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabelAlias {
    en: Vec<String>,
    zh: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Objects {
    label_alias: HashMap<String, LabelAlias>,
}

#[once(time = 3600, result = true)]
#[instrument]
async fn get_objects() -> Result<Objects> {
    Ok(serde_yaml::from_slice(
        &reqwest::get("https://github.com/QIN2DIM/hcaptcha-challenger/raw/main/src/objects.yaml")
            .await?
            .bytes()
            .await?
            .to_vec(),
    )?)
}

#[cached(result = true)]
#[instrument]
async fn get_model(name: String) -> Result<Vec<u8>> {
    Ok(reqwest::get(format!(
        "https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{name}.onnx"
    ))
    .await?
    .error_for_status()?
    .bytes()
    .await?
    .to_vec())
}

#[instrument]
async fn get_image(id: &str) -> Result<Vec<f32>> {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("sec_ch_ua"),
        HeaderValue::from_str(
            r#""Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111""#,
        )?,
    );
    headers.insert(
        HeaderName::from_static("sec-ch-ua-mobile"),
        HeaderValue::from_str("?0")?,
    );
    headers.insert(
        HeaderName::from_static("sec-ch-ua-platform"),
        HeaderValue::from_str(r#""Windows"""#)?,
    );
    headers.insert(
        HeaderName::from_static("sec-fetch-dest"),
        HeaderValue::from_str("image")?,
    );
    headers.insert(
        HeaderName::from_static("sec-fetch-mode"),
        HeaderValue::from_str("no-cors")?,
    );
    headers.insert(
        HeaderName::from_static("sec-fetch-site"),
        HeaderValue::from_str("same-site")?,
    );
    headers.insert(
        HeaderName::from_static("accept"),
        HeaderValue::from_str("image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")?,
    );
    headers.insert(
        HeaderName::from_static("accept-encoding"),
        HeaderValue::from_str("gzip, deflate, br")?,
    );
    headers.insert(
        HeaderName::from_static("accept-language"),
        HeaderValue::from_str("en-US,en;q=0.9,es;q=0.8")?,
    );
    let image = reqwest::ClientBuilder::new().default_headers(headers).user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36").build()?.get(format!("https://imgs.hcaptcha.com/{id}")).send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    Ok(tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
        let mut image = image::load_from_memory(&image.to_vec())?;
        image = image.resize_exact(64, 64, FilterType::Nearest);
        let (mut r, g, b) = image
            .as_rgb8()
            .ok_or_else(|| anyhow!("failed to convert image to rgb8"))?
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
    .await??)
}

#[instrument]
async fn check_image(model: &[u8], image_id: &str) -> Result<bool> {
    let environment = Arc::new(Environment::builder().build()?);
    let session = SessionBuilder::new(&environment)?.with_model_from_memory(&model)?;
    let image = get_image(image_id).await?;
    let input = InputTensor::FloatTensor(ArrayBase::from_shape_vec(
        IxDynImpl::from(vec![1_usize, 3, 64, 64]),
        image,
    )?);
    let output: OrtOwnedTensor<f32, _> = session.run([input])?[0].try_extract()?;
    output
        .view()
        .to_slice()
        .map(|s| s[0] > s[1])
        .ok_or_else(|| anyhow!(output.view().to_string()))
}

fn clean_prompt(prompt: String) -> Result<String> {
    let prompt = prompt.replace(".", "").to_lowercase();
    let mut label = prompt
        .rsplit_once("containing")
        .map(|(_, l)| l)
        .or_else(|| {
            prompt
                .split_once("select all")
                .and_then(|(_, l)| l.split_once("images"))
                .map(|(l, _)| l)
        })
        .ok_or_else(|| anyhow!("{prompt} doesn't contain `containing` or `select all`."))?
        .trim()
        .to_string();
    for (from, to) in [
        ("а", "a"),
        ("е", "e"),
        ("e", "e"),
        ("i", "i"),
        ("і", "i"),
        ("ο", "o"),
        ("с", "c"),
        ("ԁ", "d"),
        ("ѕ", "s"),
        ("һ", "h"),
        ("у", "y"),
        ("р", "p"),
        ("ϳ", "j"),
        ("ー", "一"),
        ("土", "士"),
    ] {
        label = label.replace(from, to);
    }
    Ok(label)
}

#[derive(Deserialize)]
struct Data {
    prompt: String,
    images: HashMap<String, String>,
}

async fn _check(data: Data) -> Result<serde_json::Value> {
    let label = clean_prompt(data.prompt)?;
    let model = Arc::new(get_model(label).await?);
    let mut images = data.images.into_iter().map(|(id, image)| {
        let model = model.clone();
        (
            id,
            tokio::spawn(async move { check_image(&model, &image).await }),
        )
    });
    let mut errors = HashMap::new();
    let mut trues = Vec::new();
    while let Some((id, handle)) = images.next() {
        match handle.await.map_err(|e| anyhow!(e)).and_then(|r| r) {
            Ok(true) => trues.push(id),
            Err(e) => {
                errors.insert(id, e.to_string());
            }
            _ => {}
        };
    }
    Ok(json! ({"trues": trues, "errors": errors}))
}

#[post("/", data = "<data>")]
async fn check(data: Json<Data>) -> std::result::Result<Json<serde_json::Value>, String> {
    match _check(data.into_inner()).await {
        Ok(res) => Ok(Json(res)),
        Err(e) => Err(e.to_string())
    }
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![check])
}