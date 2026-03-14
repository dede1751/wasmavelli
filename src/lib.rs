/// WASM bindings for wasmavelli — a Machiavelli card game solver.
///
/// # Example Usage (from JS)
/// ```js
/// import init, { GameState, Card, Rank, Suit } from "./wasmavelli.js";
///
/// await init();
/// const game = new GameState();
/// game.add_card(new Card(Rank.Ace, Suit.Spades));
/// game.add_card(new Card(Rank.Two, Suit.Spades));
/// game.add_card(new Card(Rank.Three, Suit.Spades));
/// game.add_card(Card.joker());
/// const solution = game.solve(); // returns Solution with JS-serializable groups
/// ```
use wasm_bindgen::prelude::*;

pub mod solver;
pub mod types;

use types::{Card, Solution};

// ── GameState ────────────────────────────────────────────────────────────────

#[wasm_bindgen]
#[derive(Default)]
pub struct GameState {
    cards: Vec<Card>,
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
    }

    /// Add a card to the game state.
    pub fn add_card(&mut self, card: Card) {
        self.cards.push(card);
    }

    /// Solve the current game state. If no solution exists, returns None.
    pub fn solve(&self) -> Option<Solution> {
        solver::solve(&self.cards)
    }
}
