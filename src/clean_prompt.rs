use crate::{Error, Result};

pub fn clean_prompt(prompt: &str) -> Result<String> {
    let prompt = prompt.replace('.', "").to_lowercase();
    let mut label = prompt
        .rsplit_once("containing")
        .map(|(_, l)| l.trim().trim_start_matches('a').trim_start_matches("an"))
        .or_else(|| {
            prompt
                .split_once("select all")
                .and_then(|(_, l)| l.split_once("images"))
                .map(|(l, _)| l)
        })
        .ok_or_else(|| {
            Error::BadRequest("{prompt} doesn't contain `containing` or `select all`.".to_string())
        })?
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
    ] {
        label = label.replace(from, to);
    }
    Ok(label)
}
