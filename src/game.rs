use crate::neural_network::NeuralNetwork;
use crate::training::{TrainingSample, load_training_data, save_training_data};
use rand::seq::SliceRandom;
use std::io;

pub const PLAYER: i32 = 1;
pub const COMPUTER: i32 = -1;
pub const EMPTY: i32 = 0;

pub fn display_board(board: &Vec<i32>) {
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

pub fn check_win(board: &Vec<i32>) -> Option<i32> {
    // Sprawdzamy wiersze
    for i in 0..3 {
        if board[i * 3] != EMPTY
            && board[i * 3] == board[i * 3 + 1]
            && board[i * 3] == board[i * 3 + 2]
        {
            return Some(board[i * 3]);
        }
    }
    // Sprawdzamy kolumny
    for i in 0..3 {
        if board[i] != EMPTY && board[i] == board[i + 3] && board[i] == board[i + 6] {
            return Some(board[i]);
        }
    }
    // Sprawdzamy przekątne
    if board[0] != EMPTY && board[0] == board[4] && board[0] == board[8] {
        return Some(board[0]);
    }
    if board[2] != EMPTY && board[2] == board[4] && board[2] == board[6] {
        return Some(board[2]);
    }
    None
}

pub fn check_draw(board: &Vec<i32>) -> bool {
    board.iter().all(|&cell| cell != EMPTY)
}

pub fn minimax(mut board: Vec<i32>, depth: i32, is_maximizing: bool) -> (i32, Option<usize>) {
    if let Some(winner) = check_win(&board) {
        return (
            if winner == COMPUTER {
                10 - depth
            } else if winner == PLAYER {
                depth - 10
            } else {
                0
            },
            None,
        );
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

pub fn get_player_move(board: &mut Vec<i32>) {
    loop {
        println!("Enter field number (1-9): ");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Błąd przy odczycie");
        if let Ok(num) = input.trim().parse::<usize>() {
            if num >= 1 && num <= 9 {
                let index = num - 1;
                if board[index] == EMPTY {
                    board[index] = PLAYER;
                    break;
                } else {
                    println!("To pole jest już zajęte, wybierz inne.");
                }
            } else {
                println!("Niepoprawny numer pola.");
            }
        } else {
            println!("Wprowadź liczbę.");
        }
    }
}

pub fn can_win_or_block(board: &[f32], player: f32) -> Option<usize> {
    let winning_patterns = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8], // wiersze
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8], // kolumny
        [0, 4, 8],
        [2, 4, 6], // przekątne
    ];

    for pattern in winning_patterns.iter() {
        let positions = [board[pattern[0]], board[pattern[1]], board[pattern[2]]];
        let player_count = positions.iter().filter(|&&x| x == player).count();
        let empty_count = positions.iter().filter(|&&x| x == 0.0).count();

        if player_count == 2 && empty_count == 1 {
            for &pos in pattern.iter() {
                if board[pos] == 0.0 {
                    return Some(pos);
                }
            }
        }
    }
    None
}

pub fn run() {
    let mut board = vec![EMPTY; 9];
    let mut nn = NeuralNetwork::new();

    let mut training_data = load_training_data();
    let mut played_games: Vec<TrainingSample> = Vec::new();

    println!("Trenowanie sieci neuronowej...");
    let training_samples: Vec<(Vec<f32>, Vec<f32>)> = training_data
        .iter()
        .map(|s| (s.board.clone(), s.best_move.clone()))
        .collect();
    nn.train(&training_samples, 0.01, 8000);

    let mut player_turn = true;

    loop {
        display_board(&board);

        if let Some(winner) = check_win(&board) {
            println!(
                "{}",
                if winner == PLAYER {
                    "Wygrałeś!"
                } else {
                    "Komputer wygrał!"
                }
            );

            if !played_games.is_empty() {
                training_data.extend(played_games);
                save_training_data(&training_data);
                println!("Dane treningowe zapisane do training_data.json");
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
                println!("AI wykonało zwycięski ruch w polu {}", win_move + 1);
            } else if let Some(block_move) = can_win_or_block(&board_f32, PLAYER as f32) {
                board[block_move] = COMPUTER;
                played_games.push(TrainingSample {
                    board: board_f32.clone(),
                    best_move: {
                        let mut moves = vec![0.0; 9];
                        moves[block_move] = 1.0;
                        moves
                    },
                });
                println!(
                    "AI zablokowało gracza w polu {}",
                    block_move + 1
                );
            } else {
                let (_, best_move) = minimax(board.clone(), 0, true);
                if let Some(index) = best_move {
                    board[index] = COMPUTER;
                    println!("AI wykonało strategiczny ruch w polu {}", index + 1);
                }
            }
        }

        player_turn = !player_turn;
    }
}
