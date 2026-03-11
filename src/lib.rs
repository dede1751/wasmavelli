/// WASM bindings for wasmavelli — a Machiavelli card game solver.
///
/// # Example Usage (from JS)
/// ```js
/// import init, { GameState, Rank, Suit } from "./wasmavelli.js";
///
/// await init();
/// const game = new GameState();
/// game.add_card(Rank.Ace, Suit.Spades);
/// game.add_card(Rank.Two, Suit.Spades);
/// game.add_card(Rank.Three, Suit.Spades);
/// game.add_joker();
/// const solution = game.solve(); // returns Solution with JS-serializable groups
/// ```
use wasm_bindgen::prelude::*;

pub mod solver;
pub mod types;

use types::{Card, Rank, Solution, Suit};

// ── GameState ────────────────────────────────────────────────────────────────

#[wasm_bindgen]
#[derive(Default)]
pub struct GameState {
    cards: Vec<Card>,
    jokers: usize,
}

#[wasm_bindgen]
impl GameState {
    /// Create a new empty game state.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all cards and jokers from the game state.
    pub fn clear(&mut self) {
        self.cards.clear();
        self.jokers = 0;
    }

    /// Add a normal card to the game state.
    pub fn add_card(&mut self, rank: Rank, suit: Suit) {
        self.cards.push(Card::new(rank, suit));
    }

    /// Add a joker to the game state.
    pub fn add_joker(&mut self) {
        self.jokers += 1;
    }

    /// Solve the current game state. If no solution exists, returns None.
    pub fn solve(&self) -> Option<Solution> {
        solver::solve(&self.cards, self.jokers)
    }
}
