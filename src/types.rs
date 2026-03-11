use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Rank {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
}

pub const RANKS: [Rank; 13] = [
    Rank::Ace,
    Rank::Two,
    Rank::Three,
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Eight,
    Rank::Nine,
    Rank::Ten,
    Rank::Jack,
    Rank::Queen,
    Rank::King,
];

impl Rank {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Self {
        RANKS[i]
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Suit {
    Spades,
    Clubs,
    Diamonds,
    Hearts,
}

pub const SUITS: [Suit; 4] = [Suit::Spades, Suit::Clubs, Suit::Diamonds, Suit::Hearts];

impl Suit {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Self {
        SUITS[i]
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

#[wasm_bindgen]
impl Card {
    #[wasm_bindgen(constructor)]
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    #[wasm_bindgen(js_name = asObject)]
    pub fn as_object(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(self).map_err(|e| e.into())
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Group {
    pub jokers: usize,
    #[wasm_bindgen(getter_with_clone)]
    pub cards: Vec<Card>,
}

#[wasm_bindgen]
impl Group {
    #[wasm_bindgen(js_name = asObject)]
    pub fn as_object(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(self).map_err(|e| e.into())
    }
}

#[wasm_bindgen]
#[derive(Default, Debug, Clone, Serialize)]
pub struct Solution {
    #[wasm_bindgen(getter_with_clone)]
    groups: Vec<Group>,
}

#[wasm_bindgen]
impl Solution {
    /// Serialize the full solution to a JS object (array of Group objects).
    #[wasm_bindgen(js_name = asObject)]
    pub fn as_object(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.groups).map_err(|e| e.into())
    }
}

impl Solution {
    /// Add a group (set/sequence) to the solution.
    pub fn add_group(&mut self, group: Group) {
        self.groups.push(group);
    }

    /// Access groups from Rust.
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// Total number of cards (non-joker) placed across all groups.
    pub fn total_cards(&self) -> usize {
        self.groups.iter().map(|g| g.cards.len()).sum()
    }

    /// Total number of jokers used across all groups.
    pub fn total_jokers(&self) -> usize {
        self.groups.iter().map(|g| g.jokers).sum()
    }
}
