use std::fs::File;
use std::io::Write;

use minify_html::Cfg;

fn main() {
	let markdown_as_html = ureq::post("https://api.github.com/markdown")
		.set("Accept", "application/vnd.github+json")
		.set("X-GitHub-Api-Version", "2022-11-28")
		.send(std::io::Cursor::new(
			format!(r#"{{"text": {:?}}}"#, include_str!("README.md")).into_bytes(),
		))
		.unwrap()
		.into_string()
		.unwrap();
	File::create("src/index.html")
		.unwrap()
		.write_all(&minify_html::minify(
			include_str!("src/index.html.template")
				.replace("MARKDOWN", &markdown_as_html)
				.as_bytes(),
			&Cfg {
				minify_css: true,
				..Default::default()
			},
		))
		.unwrap();
}
