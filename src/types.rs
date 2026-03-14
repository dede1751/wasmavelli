use std::fmt;

use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Rank {
    Ace = 0,
    Two = 1,
    Three = 2,
    Four = 3,
    Five = 4,
    Six = 5,
    Seven = 6,
    Eight = 7,
    Nine = 8,
    Ten = 9,
    Jack = 10,
    Queen = 11,
    King = 12,
    Joker = 14, // We leave 13 to Ace-High
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
    pub const ACE_HIGH: usize = 13;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Self {
        if i == Self::ACE_HIGH {
            Rank::Ace
        } else {
            RANKS[i]
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rank::Ace => write!(f, "A"),
            Rank::Two => write!(f, "2"),
            Rank::Three => write!(f, "3"),
            Rank::Four => write!(f, "4"),
            Rank::Five => write!(f, "5"),
            Rank::Six => write!(f, "6"),
            Rank::Seven => write!(f, "7"),
            Rank::Eight => write!(f, "8"),
            Rank::Nine => write!(f, "9"),
            Rank::Ten => write!(f, "10"),
            Rank::Jack => write!(f, "J"),
            Rank::Queen => write!(f, "Q"),
            Rank::King => write!(f, "K"),
            Rank::Joker => write!(f, "\u{1F0CF}"),
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Suit {
    Spades,
    Clubs,
    Diamonds,
    Hearts,
    Joker,
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

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Suit::Spades => write!(f, "\u{2660}"),
            Suit::Clubs => write!(f, "\u{2663}"),
            Suit::Diamonds => write!(f, "\u{2666}"),
            Suit::Hearts => write!(f, "\u{2665}"),
            Suit::Joker => write!(f, "\u{1F0CF}"),
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub const JOKER: Card = Card {
        rank: Rank::Joker,
        suit: Suit::Joker,
    };
}

#[wasm_bindgen]
impl Card {
    #[wasm_bindgen(constructor)]
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    pub fn joker() -> Self {
        Self {
            rank: Rank::Joker,
            suit: Suit::Joker,
        }
    }

    #[wasm_bindgen(js_name = asObject)]
    pub fn as_object(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(self).map_err(|e| e.into())
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Card::JOKER {
            write!(f, "\u{1F0CF}")
        } else {
            write!(f, "{}{}", self.rank, self.suit)
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Group {
    #[wasm_bindgen(getter_with_clone)]
    pub cards: Vec<Card>,
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, card) in self.cards.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{card}")?;
        }
        write!(f, "]")
    }
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

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, group) in self.groups.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{group}")?;
        }
        Ok(())
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
}
