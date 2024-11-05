use rand::{thread_rng, Rng};
use std::vec;
use rand_distr::{Distribution, Normal};
use std::f64::EPSILON;

static DEFAULT_LEARNING_RATE: f64 = 0.3f64;
static DEFAULT_MOMENTUM: f64 = 0f64;
static DEFAULT_EPOCHS: u32 = 1000;
use HaltCondition::{Epochs, MSE};

pub enum HaltCondition {
    Epochs(u32),
    MSE(f64),
}
pub struct Trainer<'a> {
    examples: &'a [(Vec<f64>, Vec<f64>)],
    learning_rate: f64,
    momentum: f64,
    halt_condition: HaltCondition,
    log_interval: Option<u32>,
    prev_weights_delta: Vec<Vec<Vec<f64>>>,
    prev_biases_delta: Vec<Vec<f64>>,
    nn: &'a mut NN,
}

impl<'a> Trainer<'a> {
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a> {
        match halt_condition {
            Epochs(epochs) if epochs < 1 => {
                panic!("must train for at least one epoch")
            }
            MSE(mse) if mse <= 0f64 => {
                panic!("MSE must be greater than 0")
            }
            _ => (),
        }

        self.halt_condition = halt_condition;
        self
    }

    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a> {
        if rate <= 0f64 {
            panic!("Learning rate can't be <= 0!");
        }

        self.learning_rate = rate;
        self
    }

    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a> {
        if momentum < 0f64 {
            panic!("Momentum can't be < 0!");
        }

        self.momentum = momentum;
        self
    }

    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => (),
        }

        self.log_interval = log_interval;
        self
    }


    fn get_cross_entropy_loss(&self) -> f64 {
        let (total_error, count) =
            self.examples
                .iter()
                .fold((0.0, 0), |(acc_error, acc_count), example| {
                    let (inputs, expected_outputs) = example;
                    let (_, outputs) = self.nn.forward(inputs);
                    let output = outputs.last().unwrap();

                    // Calculate cross-entropy error for this example
                    let error: f64 = expected_outputs
                        .iter()
                        .zip(output)
                        .map(|(expected, actual)| {
                            let actual_clamped = actual.clamp(EPSILON, 1.0 - EPSILON);
                            -expected * actual_clamped.ln() - (1.0 - expected) * (1.0 - actual_clamped).ln()
                        })
                        .sum();

                    (acc_error + error, acc_count + expected_outputs.len())
                });

        // Return the average cross-entropy loss
        total_error / count as f64
    }


    
    fn get_mean_squared_error(&self) -> f64 {
        let (total_error, count) =
            self.examples
                .iter()
                .fold((0.0, 0), |(acc_error, acc_count), example| {
                    let (inputs, expected_outputs) = example;
                    let (_, outputs) = self.nn.forward(inputs);
                    let output = outputs.last().unwrap();

                    let error: f64 = expected_outputs
                        .iter()
                        .zip(output)
                        .map(|(expected, actual)| (expected - actual).powi(2))
                        .sum();

                    (acc_error + error, acc_count + expected_outputs.len())
                });

        total_error / count as f64
    }

    pub fn go(&mut self) {
        let mut current_error = 0.0;
        let mut _epochs = 0;
        loop {
            let last_error = current_error;

            for example in self.examples {
                let (inputs, _) = example;
                let network_data = self.nn.forward(inputs);
                // println!("Network Data: {:?}", network_data.1.last().unwrap());
                self.nn.backward(
                    &example,
                    network_data,
                    self.learning_rate,
                    self.momentum,
                    &mut self.prev_weights_delta,
                    &mut self.prev_biases_delta,
                );
            }

            current_error = self.get_cross_entropy_loss();
            let error = (last_error - current_error).abs();
            _epochs += 1;

            match self.log_interval {
                Some(interval) if _epochs % interval == 0 => {
                    println!("Epochs:{:?}; Error:{:?} ", _epochs, current_error);
                }
                _ => (),
            }
            match self.halt_condition {
                Epochs(epochs_halt) => {
                    if _epochs == epochs_halt {
                        break;
                    }
                }
                MSE(target_error) => {
                    println!("Error: {:?}", error / target_error);
                    if error <= target_error {
                        break;
                    }
                }
            }
        }
    }
}

pub struct NN {
    layers: Vec<u32>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl NN {
    pub fn run(&self, data: &Vec<f64>) -> Vec<f64> {
        let (_, outputs) = self.forward(data);
        outputs.last().unwrap().clone()
    }

    fn compute_layer_input(inputs: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .zip(biases)
            .map(|(w, &b)| Self::dot_product(inputs, w).unwrap_or(0.0) - b)
            .collect()
    }

    fn forward(&self, data: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let input = data;
        let num_layers = self.layers.len();

        let mut neuron_outputs: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
        let mut neuron_inputs: Vec<Vec<f64>> = Vec::with_capacity(num_layers);

        let initial_inputs = Self::compute_layer_input(&input, &self.weights[0], &self.biases[0]);
        let initial_outputs = initial_inputs.iter().map(|&x| Self::sigmoid(x)).collect();

        neuron_inputs.push(initial_inputs);
        neuron_outputs.push(initial_outputs);

        for layer_index in 1..num_layers - 1 {
            let previous_output = &neuron_outputs[layer_index - 1];
            let layer_inputs = Self::compute_layer_input(
                previous_output,
                &self.weights[layer_index],
                &self.biases[layer_index],
            );
            let layer_outputs = layer_inputs.iter().map(|&x| Self::sigmoid(x)).collect();

            neuron_inputs.push(layer_inputs);
            neuron_outputs.push(layer_outputs);
        }

        (neuron_inputs, neuron_outputs)
    }

    fn compute_output_delta(last_output: &Vec<f64>, outputs: &Vec<f64>) -> Vec<f64> {
        last_output
            .iter()
            .zip(outputs)
            .map(|(&net_output, &desired_output)| {
                (desired_output - net_output) * net_output * (1.0 - net_output)
            })
            .collect()
    }

    fn update_weights_and_biases(
        weights: &mut Vec<Vec<f64>>,
        biases: &mut Vec<f64>,
        inputs: &Vec<f64>,
        delta: &Vec<f64>,
        learning_rate: f64,
        momentum: f64,
        prev_weights_delta: &mut Vec<Vec<f64>>,
        prev_biases_delta: &mut Vec<f64>,
    ) {
        for (i, weight_vec) in weights.iter_mut().enumerate() {
            for ((w, &input), u) in weight_vec
                .iter_mut()
                .zip(inputs)
                .zip(prev_weights_delta[i].iter_mut())
            {
                let delta_w = learning_rate * delta[i] * input + momentum * *u;
                *w += delta_w;
                *u = delta_w;
            }

            let delta_b = learning_rate * delta[i] + momentum * prev_biases_delta[i];
            biases[i] -= delta_b;
            prev_biases_delta[i] = delta_b;
        }
    }

    fn compute_hidden_delta(
        next_layer_delta: &Vec<f64>,
        cur_weights: &Vec<Vec<f64>>,
        cur_output: &Vec<f64>,
    ) -> Vec<f64> {
        cur_output
            .iter()
            .enumerate()
            .map(|(j, &output)| {
                let sum: f64 = next_layer_delta
                    .iter()
                    .enumerate()
                    .map(|(k, &delta)| delta * cur_weights[k][j])
                    .sum();
                sum * output * (1.0 - output)
            })
            .collect()
    }

    fn backward(
        &mut self,
        data: &(Vec<f64>, Vec<f64>),
        network_data: (Vec<Vec<f64>>, Vec<Vec<f64>>),
        learning_rate: f64,
        momentum: f64,
        prev_weights_delta: &mut Vec<Vec<Vec<f64>>>,
        prev_biases_delta: &mut Vec<Vec<f64>>,
    ) {
        let (inputs, outputs) = data;
        let num_layers = self.layers.len();
        let (_, neuron_outputs) = network_data;
        let last_output = neuron_outputs.last().unwrap();
        let mut deltas: Vec<Vec<f64>> = vec![Self::compute_output_delta(last_output, outputs)];

        for index in (2..num_layers).rev() {
            let delta_index = num_layers - index - 1;

            Self::update_weights_and_biases(
                &mut self.weights[index - 1],
                &mut self.biases[index - 1],
                &neuron_outputs[index - 2],
                &deltas[delta_index],
                learning_rate,
                momentum,
                &mut prev_weights_delta[index - 1],
                &mut prev_biases_delta[index - 1],
            );

            let current_delta = Self::compute_hidden_delta(
                &deltas[delta_index],
                &self.weights[index - 1],
                &neuron_outputs[index - 2],
            );
            deltas.push(current_delta);
        }

        Self::update_weights_and_biases(
            &mut self.weights[0],
            &mut self.biases[0],
            inputs,
            &deltas.last().unwrap(),
            learning_rate,
            momentum,
            &mut prev_weights_delta[0],
            &mut prev_biases_delta[0],
        );
    }

    pub fn new(layers: &Vec<u32>) -> NN {
        if layers.is_empty() {
            panic!("Layers cannot be empty");
        }

        let mut nn = NN {
            layers: layers.clone(),
            weights: vec![],
            biases: vec![],
        };

        nn.generate_weights();
        nn
    }

    fn generate_weights(&mut self) {

        let mut rng = thread_rng();
        
        self.weights = self.layers.windows(2).map(|layer_pair| {
            let fan_in = layer_pair[0] as f64;
            let he_std_dev = (2.0 / fan_in).sqrt();
            let normal_dist = Normal::new(0.0, he_std_dev).unwrap();
            
            (0..layer_pair[1])
                .map(|_| {
                    (0..layer_pair[0])
                        .map(|_| normal_dist.sample(&mut rng))
                        .collect()
                })
                .collect()
        }).collect();
    
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        self.biases = self.layers.iter().skip(1).map(|&l| {
            (0..l).map(|_| normal_dist.sample(&mut rng)).collect()
        }).collect();
    }

    pub fn train<'a>(&'a mut self, examples: &'a [(Vec<f64>, Vec<f64>)]) -> Trainer {
        let output_layer_size = *self.layers.last().unwrap() as usize;
        for example in examples {
            let (input, output) = example;
            if input.len() != self.layers[0] as usize {
                panic!("Input vector must have the same length as input layer.");
            }
            if output.len() != output_layer_size {
                panic!("Output vector must have the same length as output layer.");
            }
        }

        Trainer {
            examples: &examples,
            momentum: DEFAULT_MOMENTUM,
            prev_weights_delta: self
                .weights
                .iter()
                .map(|layer| vec![vec![0.0; layer[0].len()]; layer.len()])
                .collect(),
            prev_biases_delta: self
                .biases
                .iter()
                .map(|layer| vec![0.0; layer.len()])
                .collect(),
            learning_rate: DEFAULT_LEARNING_RATE,
            halt_condition: Epochs(DEFAULT_EPOCHS),
            log_interval: None,
            nn: self,
        }
    }

    fn dot_product(first: &[f64], second: &[f64]) -> Result<f64, &'static str> {
        if first.len() != second.len() {
            return Err("Must have the same number of elements");
        }

        let result = first.iter().zip(second).map(|(x, y)| x * y).sum();
        Ok(result)
    }

    fn sigmoid(y: f64) -> f64 {
        1f64 / (1f64 + (-y).exp())
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
    
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}