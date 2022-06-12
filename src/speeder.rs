use anyhow::Context;
use clap::{arg, Command};
use hound::{WavReader, WavWriter};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

fn main() -> Result<(), anyhow::Error> {
    let matches = Command::new("speeder")
        .about("Dilate audio speed")
        .arg(arg!(<INPUT> "input audio file (only wav files are currently supported)"))
        .arg(arg!(<OUTPUT> "output audio file (only wav files are currently supported)"))
        .arg(
            arg!(--factor "the factor by which to multiply the speed")
                .takes_value(true)
                .required(true)
                .validator(|s| s.parse::<f32>()),
        )
        .get_matches();

    let factor = matches
        .value_of_t::<f32>("factor")
        .unwrap()
        .clamp(1.0 / 16.0, 16.0);
    let in_name = matches.value_of("INPUT").unwrap();
    let out_name = matches.value_of("OUTPUT").unwrap();
    let in_file = BufReader::new(
        File::open(in_name)
            .with_context(|| format!("Failed to open input file \"{}\"", in_name))?,
    );
    let out_file = BufWriter::new(
        File::create(out_name)
            .with_context(|| format!("Failed to open output file \"{}\"", out_name))?,
    );

    let reader = WavReader::new(in_file)?;
    let sample_rate = reader.spec().sample_rate;
    let mut writer = WavWriter::new(out_file, reader.spec())?;
    // FIXME: channels, different bit sizes
    let samples = reader.into_samples::<i16>().map(move |s| s.unwrap());

    let stream = speeder::Stream::new(samples, sample_rate, factor);
    for s in stream {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}
