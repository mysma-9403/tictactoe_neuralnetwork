use std::fs;
use std::io::{self, Write};
use rand::Rng;
use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;

const PLAYER: i32 = 1;
const COMPUTER: i32 = -1;
const EMPTY: i32 = 0;

/// TiacTacToe NeuralNetwork
struct NeuralNetwork {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct TrainingSample {
    board: Vec<f32>,
    best_move: Vec<f32>,
}

fn minimax(mut board: Vec<i32>, depth: i32, is_maximizing: bool) -> (i32, Option<usize>) {
    if let Some(winner) = check_win(&board) {
        return (if winner == COMPUTER { 10 - depth } else if winner == PLAYER { depth - 10 } else { 0 }, None);
    }

    if check_draw(&board) {
        return (0, None);
    }

    let mut best_score = if is_maximizing { i32::MIN } else { i32::MAX };
    let mut best_move = None;

    for i in 0..9 {
        if board[i] == EMPTY {
            board[i] = if is_maximizing { COMPUTER } else { PLAYER };
            let (score, _) = minimax(board.clone(), depth + 1, !is_maximizing);
            board[i] = EMPTY;

            if is_maximizing {
                if score > best_score {
                    best_score = score;
                    best_move = Some(i);
                }
            } else {
                if score < best_score {
                    best_score = score;
                    best_move = Some(i);
                }
            }
        }
    }
    (best_score, best_move)
}


fn save_training_data(training_data: &Vec<TrainingSample>) {
    match serde_json::to_string(training_data) {
        Ok(json) => {
            if let Err(e) = fs::write("training_data.json", json) {
                println!("❌ Błąd zapisu pliku: {}", e);
            } else {
                println!("✅ Dane zapisane do pliku training_data.json");
            }
        }
        Err(e) => println!("❌ Błąd serializacji JSON: {}", e),
    }
}

fn load_training_data() -> Vec<TrainingSample> {
    if let Ok(data) = fs::read_to_string("training_data.json") {
        if let Ok(training_data) = serde_json::from_str::<Vec<TrainingSample>>(&data) {
            return training_data;
        }
    }
    vec![]
}
impl NeuralNetwork {
    fn new() -> Self {
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

    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    fn tanh_derivative(x: f32) -> f32 {
        1.0 - x.powi(2)
    }

    fn forward(&self, input: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
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

    /// Funkcja trenująca sieć neuronową
    fn train(&mut self, training_data: &Vec<(Vec<f32>, Vec<f32>)>, learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, expected_output) in training_data.iter() {
                let (hidden, output) = self.forward(input);

                let mut output_error = vec![0.0; 9];
                let mut sample_loss = 0.0;

                for i in 0..9 {
                    let error = expected_output[i] - output[i];
                    output_error[i] = error * Self::tanh_derivative(output[i]);
                    sample_loss += error.powi(2); // Obliczamy sumę błędów dla tej próbki
                }
                total_loss += sample_loss / 9.0; // Uśredniamy dla każdej próbki

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

            // Wyświetlamy postęp co 100 epok
            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoka: {:4} | Średni błąd: {:.6} | Learning rate: {}",
                         epoch, total_loss / training_data.len() as f32, learning_rate);
            }
        }
    }
}

fn display_board(board: &Vec<i32>) {
    println!("-------------");
    for i in 0..3 {
        print!("|");
        for j in 0..3 {
            let cell = board[i * 3 + j];
            let symbol = match cell {
                PLAYER => "X",
                COMPUTER => "O",
                _ => " ",
            };
            print!(" {} |", symbol);
        }
        println!("\n-------------");
    }
}

fn check_win(board: &Vec<i32>) -> Option<i32> {
    for i in 0..3 {
        if board[i * 3] != EMPTY && board[i * 3] == board[i * 3 + 1] && board[i * 3] == board[i * 3 + 2] {
            return Some(board[i * 3]);
        }
    }
    for i in 0..3 {
        if board[i] != EMPTY && board[i] == board[i + 3] && board[i] == board[i + 6] {
            return Some(board[i]);
        }
    }
    if board[0] != EMPTY && board[0] == board[4] && board[0] == board[8] {
        return Some(board[0]);
    }
    if board[2] != EMPTY && board[2] == board[4] && board[2] == board[6] {
        return Some(board[2]);
    }
    None
}

fn check_draw(board: &Vec<i32>) -> bool {
    board.iter().all(|&cell| cell != EMPTY)
}

fn get_player_move(board: &mut Vec<i32>) {
    loop {
        println!("Podaj numer pola (1-9): ");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Błąd odczytu");
        if let Ok(num) = input.trim().parse::<usize>() {
            if num >= 1 && num <= 9 {
                let index = num - 1;
                if board[index] == EMPTY {
                    board[index] = PLAYER;
                    break;
                } else {
                    println!("Pole zajęte, wybierz inne.");
                }
            } else {
                println!("Nieprawidłowy numer pola.");
            }
        } else {
            println!("Wprowadź liczbę.");
        }
    }
}

fn get_computer_move(board: &mut Vec<i32>, nn: &NeuralNetwork) {
    let input: Vec<f32> = board
        .iter()
        .map(|&cell| {
            if cell == PLAYER {
                1.0
            } else if cell == COMPUTER {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let output = nn.forward(&input);

    let mut best_move = None;
    let mut best_value = -100.0;
    let (_, output) = nn.forward(&input); // Pobieramy tylko output
    for (i, &value) in output.iter().enumerate() {
        if board[i] == EMPTY && value > best_value {
            best_value = value;
            best_move = Some(i);
        }
    }

    if let Some(index) = best_move {
        board[index] = COMPUTER;
        println!("Komputer wybiera pole {}", index + 1);
    }
}

fn can_win_or_block(board: &[f32], player: f32) -> Option<usize> {
    let winning_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // ✅ Wiersze (poziomo)
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // ✅ Kolumny (pionowo)
        [0, 4, 8], [2, 4, 6]  // ✅ Przekątne
    ];

    for &pattern in &winning_patterns {
        let positions = [board[pattern[0]], board[pattern[1]], board[pattern[2]]];

        println!("{:?}", positions);
        // ✅ Liczymy liczbę pól zajętych przez gracza i liczbę pustych pól
        let player_count = positions.iter().filter(|&&x| x == player).count();
        let empty_count = positions.iter().filter(|&&x| x == 0.0).count();

        // ✅ Muszą być dokładnie **dwa pola gracza** i jedno wolne
        if player_count == 2 && empty_count == 1 {
            for &pos in &pattern {
                println!("{}", pos);
                println!("{}", board[pos]);
                if board[pos] == 0.0 {
                    return Some(pos); // **Zwracamy tylko JEDYNE wolne pole w tej linii**
                }
            }
        }
    }
    None
}




/// Generuje lepsze dane treningowe
fn generate_training_data(num_samples: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    let mut training_data = Vec::new();

    for _ in 0..num_samples {
        let mut board: Vec<f32> = vec![0.0; 9];

        // Dodajemy kilka ruchów, aby plansza nie była pusta
        for _ in 0..rng.gen_range(2..6) {
            let empty_positions: Vec<usize> = board.iter().enumerate()
                .filter(|&(_, &v)| v == 0.0)
                .map(|(i, _)| i)
                .collect();

            if let Some(&pos) = empty_positions.choose(&mut rng) {
                board[pos] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 }; // X lub O
            }
        }

        let mut best_moves = vec![0.0; 9];

        // Priorytet 1: Jeśli komputer może wygrać, to to zrobi
        if let Some(win_move) = can_win_or_block(&board, -1.0) {
            best_moves[win_move] = 1.0;
        }
        // Priorytet 2: Jeśli gracz może wygrać, komputer go blokuje
        else if let Some(block_move) = can_win_or_block(&board, 1.0) {
            best_moves[block_move] = 1.0;
        }
        // Priorytet 3: Wybieramy losowy ruch z pustych pól
        else {
            let empty_positions: Vec<usize> = board.iter().enumerate()
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

fn main() {
    let mut board = vec![EMPTY; 9];
    let mut nn = NeuralNetwork::new();

    let mut training_data = load_training_data();
    let mut played_games: Vec<TrainingSample> = vec![];

    println!("Trenuję sieć...");
    nn.train(
        &training_data.iter().map(|s| (s.board.clone(), s.best_move.clone())).collect(),
        0.01,
        8000,
    );

    let mut player_turn = true;

    loop {
        display_board(&board);

        if let Some(winner) = check_win(&board) {
            println!("{}", if winner == PLAYER { "Wygrałeś!" } else { "Komputer wygrał!" });

            // ✅ Zapisujemy tylko blokady do bazy treningowej
            if !played_games.is_empty() {
                training_data.extend(played_games);
                save_training_data(&training_data);
                println!("✅ Zapisano sytuacje blokowania do training_data.json");
            }

            break;
        }

        if check_draw(&board) {
            println!("Remis!");
            break;
        }

        if player_turn {
            get_player_move(&mut board);
        } else {
            let board_f32: Vec<f32> = board.iter().map(|&x| x as f32).collect();

            if let Some(win_move) = can_win_or_block(&board_f32, COMPUTER as f32) {
                board[win_move] = COMPUTER;
                println!("🏆 AI wykonało ruch wygrywający na polu {}", win_move + 1);
            } else if let Some(block_move) = can_win_or_block(&board_f32, PLAYER as f32) {
                board[block_move] = COMPUTER;
                played_games.push(TrainingSample { board: board_f32.clone(), best_move: vec![0.0; 9] });
                played_games.last_mut().unwrap().best_move[block_move] = 1.0;
                println!("✅ AI zablokowało gracza na polu {}", block_move + 1);
            } else {
                let (_, best_move) = minimax(board.clone(), 0, true);
                if let Some(index) = best_move {
                    board[index] = COMPUTER;
                    println!("🤖 AI wykonało strategiczny ruch na polu {}", index + 1);
                }
            }
        }



        player_turn = !player_turn;
    }
}

