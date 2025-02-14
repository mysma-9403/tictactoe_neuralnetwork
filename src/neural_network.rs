use rand::Rng;

pub struct NeuralNetwork {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let w1: Vec<Vec<f32>> = (0..18)
            .map(|_| (0..9).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();
        let b1: Vec<f32> = (0..18).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let w2: Vec<Vec<f32>> = (0..9)
            .map(|_| (0..18).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();
        let b2: Vec<f32> = (0..9).map(|_| rng.gen_range(-0.5..0.5)).collect();

        NeuralNetwork { w1, b1, w2, b2 }
    }

    pub fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    pub fn tanh_derivative(x: f32) -> f32 {
        1.0 - x.powi(2)
    }

    pub fn forward(&self, input: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = vec![0.0; 18];
        for i in 0..18 {
            hidden[i] = self.b1[i];
            for j in 0..9 {
                hidden[i] += self.w1[i][j] * input[j];
            }
            hidden[i] = Self::tanh(hidden[i]);
        }

        let mut output = vec![0.0; 9];
        for i in 0..9 {
            output[i] = self.b2[i];
            for j in 0..18 {
                output[i] += self.w2[i][j] * hidden[j];
            }
            output[i] = Self::tanh(output[i]);
        }

        (hidden, output)
    }

    pub fn train(
        &mut self,
        training_data: &Vec<(Vec<f32>, Vec<f32>)>,
        learning_rate: f32,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, expected_output) in training_data.iter() {
                let (hidden, output) = self.forward(input);

                let mut output_error = vec![0.0; 9];
                let mut sample_loss = 0.0;

                for i in 0..9 {
                    let error = expected_output[i] - output[i];
                    output_error[i] = error * Self::tanh_derivative(output[i]);
                    sample_loss += error.powi(2);
                }
                total_loss += sample_loss / 9.0;

                let mut hidden_error = vec![0.0; 18];
                for i in 0..18 {
                    for j in 0..9 {
                        hidden_error[i] += output_error[j] * self.w2[j][i];
                    }
                    hidden_error[i] *= Self::tanh_derivative(hidden[i]);
                }

                for i in 0..9 {
                    for j in 0..18 {
                        self.w2[i][j] += learning_rate * output_error[i] * hidden[j];
                    }
                    self.b2[i] += learning_rate * output_error[i];
                }

                for i in 0..18 {
                    for j in 0..9 {
                        self.w1[i][j] += learning_rate * hidden_error[i] * input[j];
                    }
                    self.b1[i] += learning_rate * hidden_error[i];
                }
            }

            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch: {:4} | Average error: {:.6} | Learning rate: {}",
                    epoch,
                    total_loss / training_data.len() as f32,
                    learning_rate
                );
            }
        }
    }
}
