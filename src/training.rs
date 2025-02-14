use serde::{Deserialize, Serialize};
use std::fs;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::game::can_win_or_block;

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingSample {
    pub board: Vec<f32>,
    pub best_move: Vec<f32>,
}

pub fn save_training_data(training_data: &Vec<TrainingSample>) {
    match serde_json::to_string(training_data) {
        Ok(json) => {
            if let Err(e) = fs::write("training_data.json", json) {
                println!("❌ Błąd zapisu do pliku: {}", e);
            } else {
                println!("✅ Dane zapisane do training_data.json");
            }
        }
        Err(e) => println!("❌ Błąd serializacji JSON: {}", e),
    }
}

pub fn load_training_data() -> Vec<TrainingSample> {
    if let Ok(data) = fs::read_to_string("training_data.json") {
        if let Ok(training_data) = serde_json::from_str::<Vec<TrainingSample>>(&data) {
            return training_data;
        }
    }
    vec![]
}

pub fn generate_training_data(num_samples: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    let mut training_data = Vec::new();

    for _ in 0..num_samples {
        let mut board: Vec<f32> = vec![0.0; 9];

        // Dodaj kilka losowych ruchów, aby plansza nie była pusta
        for _ in 0..rng.gen_range(2..6) {
            let empty_positions: Vec<usize> = board
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v == 0.0)
                .map(|(i, _)| i)
                .collect();

            if let Some(&pos) = empty_positions.choose(&mut rng) {
                board[pos] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            }
        }

        let mut best_moves = vec![0.0; 9];

        // Priorytet 1: Jeżeli komputer może wygrać, wybieramy ten ruch
        if let Some(win_move) = can_win_or_block(&board, -1.0) {
            best_moves[win_move] = 1.0;
        }
        // Priorytet 2: Jeżeli gracz może wygrać, komputer blokuje
        else if let Some(block_move) = can_win_or_block(&board, 1.0) {
            best_moves[block_move] = 1.0;
        }
        // Priorytet 3: Wybieramy losowy ruch z pustych pól
        else {
            let empty_positions: Vec<usize> = board
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v == 0.0)
                .map(|(i, _)| i)
                .collect();

            if let Some(&random_move) = empty_positions.choose(&mut rng) {
                best_moves[random_move] = 1.0;
            }
        }

        training_data.push((board, best_moves));
    }

    training_data
}
