use rusty_net::{NN, HaltCondition::Epochs};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;



fn read_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<(Vec<f64>, Vec<f64>)>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let value: f64 = line?.trim().parse().expect("Invalid number format");
        data.push(value);
    }

    let mut examples = Vec::new();
    for chunk in data.chunks_exact(4) {
        let input = chunk[0..3].to_vec();
        let output = vec![chunk[3]];
        examples.push((input, output));
    }

    Ok(examples)
}

fn main() {
    
    let file_path = "./data/samples.txt";
    let examples = read_file(file_path).unwrap();
    
    let mut nn = NN::new(&vec![3,10, 1]);
    nn.train(&examples)
        .rate(0.1)
        .momentum(0.9)
        .log_interval(Some(100))
        .halt_condition(Epochs(50000))
        .go();


    let file_path = "./data/validation.txt";
    let samples = read_file(file_path).unwrap();
    let mut error_sum = 0.0;
    let mut count = 0;

    for sample in samples {
        let inputs = &sample.0;
        let result = nn.run(inputs);
        let output = result[0];
        let actual = sample.1[0];

        if actual != 0.0 {
            let relative_error = (output - actual).abs() / actual.abs();
            error_sum += relative_error;
            count += 1;
        }

        println!("{:?} == {:?}", result[0].round(), sample.1);
    }

    println!(
        "Mean Relative Error: {:.2}%",
        (error_sum / count as f64) * 100.0
    );
}
