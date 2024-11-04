use rusty_net::{NN, HaltCondition::Epochs};

fn main() {
    let mut m = NN::new(&vec![2, 5, 1]);
    
    
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];


    m.train(&examples)
        .rate(0.1)
        .momentum(0.9)
        .log_interval(Some(100))
        .halt_condition(Epochs(50000))
        .go();


    let mut error_sum = 0.0;
    let mut count = 0;

    for sample in examples {
        let inputs = &sample.0;
        let result = m.run(inputs);
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
