use std::{fs::File, io::Write};

use pandoc::{InputKind, OutputKind, PandocOption, PandocOutput};

fn main() {
    let mut pandoc = pandoc::new();
    pandoc.add_option(PandocOption::Standalone);
    pandoc.add_option(PandocOption::TitlePrefix(
        "Hcaptcha Solver Rest API".to_string(),
    ));
    pandoc.set_input(InputKind::Pipe(include_str!("README.md").to_string()));
    pandoc.set_output(OutputKind::Pipe);
    match pandoc.execute().unwrap() {
        PandocOutput::ToBuffer(s) => File::create("src/index.html")
            .unwrap()
            .write_all(s.as_bytes())
            .unwrap(),
        _ => panic!("Output not string"),
    }
}
