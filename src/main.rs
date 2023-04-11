#[macro_use]
extern crate serde;
#[macro_use]
extern crate tokio;
#[macro_use]
extern crate tracing;

use std::collections::HashMap;

use anyhow::Result;
use cached::proc_macro::{cached, once};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabelAlias {
    en: Vec<String>,
    zh: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Objects {
    label_alias: HashMap<String, LabelAlias>,
}

#[once(time = 86400, result = true)]
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
    Ok(
        reqwest::get(format!("https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{name}.onnx"))
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec())
}

#[main]
async fn main() {
    let objects = get_objects().await.unwrap();
    println!("{:?}", get_model(dbg!(objects.label_alias.keys().nth(1).unwrap().clone())).await);
}
