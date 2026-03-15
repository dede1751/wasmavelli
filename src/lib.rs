/// WASM bindings for wasmavelli — a Machiavelli card game solver.
///
/// # Example Usage (from JS)
/// ```js
/// import init, { GameState, Card, Rank, Suit } from "./wasmavelli.js";
///
/// await init();
/// const game = new GameState();
/// game.add_board_card(new Card(Rank.Ace, Suit.Spades));  // board cards must be used
/// game.add_board_card(new Card(Rank.Two, Suit.Spades));
/// game.add_hand_card(new Card(Rank.Three, Suit.Spades)); // hand cards can be left unused
/// game.add_hand_card(Card.joker());
/// game.add_hand_card(new Card(Rank.Four, Suit.Hearts));
/// const sol1 = game.solve();
/// // sol1: { groups: [A♠, 2♠, 3♠, JOKER], remaining: [4♥] } 
/// 
/// game.clear();
/// game.add_board_card(new Card(Rank.Ace, Suit.Spades));
/// const sol2 = game.solve();
/// // sol2: null
/// ```
use wasm_bindgen::prelude::*;

pub mod solver;
pub mod types;

use types::{Card, Group, Solution};

// ── GameState ────────────────────────────────────────────────────────────────

#[wasm_bindgen]
#[derive(Default)]
pub struct GameState {
    board: Group,
    hand: Group,
}

#[wasm_bindgen]
impl GameState {
    /// Create a new empty game state.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all cards from the game state.
    pub fn clear(&mut self) {
        self.board = Group::default();
        self.hand = Group::default();
    }

    /// Add a card in the player's hand.
    pub fn add_hand_card(&mut self, card: Card) {
        self.hand.cards.push(card);
    }

    /// Add a card to the game board.
    pub fn add_board_card(&mut self, card: Card) {
        self.board.cards.push(card);
    }

    /// Solve the current game state. If no solution exists, returns None.
    pub fn solve(&self) -> Option<Solution> {
        solver::solve(&self.board, &self.hand)
    }
}
