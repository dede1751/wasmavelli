use std::collections::HashMap;

use good_lp::{
    Expression, ProblemVariables, Solution, SolverModel, Variable, default_solver, solvers::microlp::MicroLpProblem, variable
};

use crate::types::{self, Card, Group, Rank, Suit, RANKS, SUITS};

const SET_MAX: usize = 4;  // max cards in a set
const SEQ_MAX: usize = 14; // max cards in a sequence (A 2 3 … K A)

type CardMat = [[usize; SET_MAX]; SEQ_MAX];

/// Compute a matrix of card counts indexed by rank and suit.
/// Index 13 mirrors index 0 (Ace) for ace-high sequence detection.
fn card_matrix(cards: &[Card]) -> CardMat {
    let mut mat: CardMat = [[0; SET_MAX]; SEQ_MAX];
    for card in cards {
        mat[card.rank.index()][card.suit.index()] += 1;
    }
    for s in 0..4 {
        mat[13][s] = mat[0][s];
    }
    mat
}

/// Enumerate all valid sequences (3-14 consecutive ranks of a single suit).
fn enumerate_sequences(mat: &CardMat) -> Vec<Group> {
    let mut groups = Vec::new();

    for suit in SUITS {
        for start in 0..(SEQ_MAX - 2) {
            if mat[start][suit.index()] == 0 {
                continue;
            }
            // Find how far this consecutive run extends
            let mut end = start + 1;
            while end < SEQ_MAX && mat[end][suit.index()] > 0 {
                end += 1;
            }
            // Add all valid-length sequences starting here
            for len in 3..=(end - start) {
                let cards = (start..start + len)
                    .map(|i| Card::new(Rank::from_index(i % 13), suit)) // wrap index for ace-high
                    .collect();
                groups.push(Group { jokers: 0, cards });
            }
        }
    }

    groups
}

/// Enumerate all valid sets (3–4 cards of the same rank, each a different suit).
fn enumerate_sets(mat: &CardMat) -> Vec<Group> {
    let mut groups = Vec::new();

    for rank in RANKS {
        let suits_present: Vec<Suit> = SUITS.iter().copied()
            .filter(|&s| mat[rank.index()][s.index()] > 0)
            .collect();
        let n = suits_present.len();
        let cards: Vec<Card> = suits_present.iter().map(|&s| Card::new(rank, s)).collect();

        // Add the set with all suits present (if 3 or 4 suits are present)
        if n >= 3 {
            groups.push(Group { jokers: 0, cards: cards.clone() });
        }

        // Add all possible subsets (if 4 suits are present)
        if n == 4 {
            for skip_idx in 0..4 {
                let mut cards_removed = cards.clone();
                cards_removed.remove(skip_idx);
                groups.push(Group { jokers: 0, cards: cards_removed });
            }
        }
    }

    groups
}

/// Compute the maximum number of times each group can be selected based on the input card counts.
fn group_max(mat: &CardMat, groups: &[Group]) -> Vec<usize> {
    groups
        .iter()
        .map(|group| {
            group
                .cards
                .iter()
                .map(|c| mat[c.rank.index()][c.suit.index()])
                .min()
                .unwrap_or(0)
        })
        .collect()
}

/// Compute a mapping from each card to the indices of groups that contain it.
fn card_to_groups(groups: &[Group]) -> HashMap<Card, Vec<usize>> {
    let mut card_to_groups: HashMap<Card, Vec<usize>> = HashMap::new();
    for (idx, group) in groups.iter().enumerate() {
        for &card in &group.cards {
            card_to_groups.entry(card).or_default().push(idx);
        }
    }
    card_to_groups
}

fn build_lp(mat: &CardMat, groups: &[Group]) -> (MicroLpProblem, Vec<Variable>) {
    let group_max = group_max(mat, groups);
    let card_to_groups = card_to_groups(groups);

    // Variables: the number of times we select each group
    let mut vars = ProblemVariables::new();
    let x: Vec<Variable> = group_max
        .iter()
        .map(|&gm| vars.add(variable().integer().min(0).max(gm as f64)))
        .collect();

    // Objective: maximise total cards placed, with a small tiebreaker favouring fewer groups
    let objective: Expression = groups
        .iter()
        .enumerate()
        .map(|(i, group)| x[i] * group.cards.len() as f64 - x[i] * (1.0 / 1024.0))
        .sum();
    let mut model = vars.maximise(objective).using(default_solver);

    // Constraints:
    //  - Each input card must not be used more times than its total occurrence in the input.
    for (&card, group_indices) in &card_to_groups {
        let lhs: Expression = group_indices.iter().map(|&idx| x[idx]).sum();
        model = model.with(lhs << mat[card.rank.index()][card.suit.index()] as f64);
    }

    (model, x)
}

pub fn solve(cards: &[Card], _jokers: usize) -> Result<types::Solution, String> {
    let mat = card_matrix(cards);

    let groups: Vec<Group> =
        [enumerate_sequences(&mat), enumerate_sets(&mat)].concat();
    if groups.is_empty() {
        return Ok(types::Solution::default());
    }

    let (model, x) = build_lp(&mat, &groups);
    let solution = model.solve().map_err(|e| e.to_string())?;

    let mut result = types::Solution::default();
    for (idx, group) in groups.into_iter().enumerate() {
        let val = solution.value(x[idx]).round() as usize;
        for _ in 0..val {
            result.add_group(group.clone());
        }
    }

    Ok(result)
}

/// Note: this test suite is entirely AI-generated.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Card, Group, Rank, Suit};
    use std::collections::HashMap;

    /// Check that a group is a valid set: 3-4 cards of same rank, all different suits.
    /// Jokers fill in for missing cards. Total size (cards + jokers) must be 3 or 4.
    fn is_valid_set(g: &Group) -> bool {
        let total = g.cards.len() + g.jokers;
        if !(3..=4).contains(&total) {
            return false;
        }
        if g.cards.is_empty() {
            return g.jokers >= 3; // all-joker group
        }
        let rank = g.cards[0].rank;
        if !g.cards.iter().all(|card| card.rank == rank) {
            return false;
        }
        // All suits must be distinct
        let mut seen = [false; 4];
        for card in &g.cards {
            let idx = card.suit.index();
            if seen[idx] {
                return false;
            }
            seen[idx] = true;
        }
        true
    }

    /// Check that a group is a valid sequence: 3+ consecutive cards of same suit.
    /// Jokers fill gaps. Total size (cards + jokers) must be >= 3.
    /// Ace can be low (before 2) or high (after King), but no wrap-around.
    fn is_valid_sequence(g: &Group) -> bool {
        let total = g.cards.len() + g.jokers;
        if total < 3 {
            return false;
        }
        if g.cards.is_empty() {
            return g.jokers >= 3; // all-joker group
        }
        let suit = g.cards[0].suit;
        if !g.cards.iter().all(|card| card.suit == suit) {
            return false;
        }
        // Get sorted rank indices
        let mut indices: Vec<usize> = g.cards.iter().map(|card| card.rank.index()).collect();
        indices.sort();
        // Check for duplicates
        for w in indices.windows(2) {
            if w[0] == w[1] {
                return false;
            }
        }
        // Try to place the cards + jokers in a consecutive run.
        // The span from min to max rank must be coverable with cards + jokers.
        let min_r = indices[0];
        let max_r = indices[indices.len() - 1];

        // Try normal ordering (Ace = 0)
        let span = max_r - min_r + 1;
        if span <= total && span >= 3 && span - g.cards.len() <= g.jokers {
            return true;
        }

        // Try Ace-high: if Ace is present, treat it as index 13
        if indices[0] == 0 {
            let mut hi_indices: Vec<usize> = indices
                .iter()
                .map(|&i| if i == 0 { 13 } else { i })
                .collect();
            hi_indices.sort();
            let min_h = hi_indices[0];
            let max_h = hi_indices[hi_indices.len() - 1];
            let span_h = max_h - min_h + 1;
            if span_h <= total && span_h >= 3 && span_h - g.cards.len() <= g.jokers {
                return true;
            }
        }

        false
    }

    /// Check that a group is valid: either a set, a sequence, or pure jokers.
    fn is_valid_group(g: &Group) -> bool {
        if g.cards.is_empty() {
            return g.jokers >= 3;
        }
        is_valid_set(g) || is_valid_sequence(g)
    }

    /// Validate that a solution is legal given the input cards and joker count.
    fn validate_solution(sol: &types::Solution, input_cards: &[Card], input_jokers: usize) {
        // 1. Every group must be valid
        for (i, g) in sol.groups().iter().enumerate() {
            assert!(
                is_valid_group(g),
                "Group {i} is not a valid set or sequence: {g:?}"
            );
        }

        // 2. Cards used must not exceed input card counts
        let mut input_counts: HashMap<Card, usize> = HashMap::new();
        for &card in input_cards {
            *input_counts.entry(card).or_default() += 1;
        }
        let mut used_counts: HashMap<Card, usize> = HashMap::new();
        for g in sol.groups() {
            for &card in &g.cards {
                *used_counts.entry(card).or_default() += 1;
            }
        }
        for (&card, &used) in &used_counts {
            let available = input_counts.get(&card).copied().unwrap_or(0);
            assert!(
                used <= available,
                "Card {card:?} used {used} times but only {available} available"
            );
        }

        // 3. Jokers used must not exceed input joker count
        let jokers_used: usize = sol.total_jokers();
        assert!(
            jokers_used <= input_jokers,
            "Used {jokers_used} jokers but only {input_jokers} available"
        );
    }

    /// Assert that the solution places exactly `n` cards (non-joker) on the table.
    fn assert_cards_placed(sol: &types::Solution, n: usize) {
        let cards = sol.total_cards();
        assert_eq!(cards, n, "Expected {n} cards placed, got {cards}");
    }

    /// Assert that the solution uses exactly `n` jokers.
    fn assert_jokers_used(sol: &types::Solution, n: usize) {
        let jokers = sol.total_jokers();
        assert_eq!(jokers, n, "Expected {n} jokers used, got {jokers}");
    }

    // ── Empty / Trivial Inputs ──────────────────────────────────────────

    #[test]
    fn empty_input() {
        let sol = solve(&[], 0).unwrap();
        validate_solution(&sol, &[], 0);
        assert_eq!(sol.groups().len(), 0);
    }

    #[test]
    fn single_card_no_jokers() {
        let cards = [Card::new(Rank::Ace, Suit::Spades)];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn two_cards_no_jokers() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    // ── Simple Sets ────────────────────────────────────────────────────

    #[test]
    fn set_of_three() {
        let cards = [
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Five, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn set_of_four() {
        let cards = [
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::King, Suit::Clubs),
            Card::new(Rank::King, Suit::Diamonds),
            Card::new(Rank::King, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 4);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn not_a_set_same_suit() {
        // Two fives of spades + one five of clubs: can only use one 5♠ per set
        let cards = [
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn set_with_leftover() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Seven, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
        assert_eq!(sol.groups().len(), 1);
    }

    // ── Simple Sequences ────────────────────────────────────────────────

    #[test]
    fn sequence_of_three() {
        let cards = [
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn sequence_of_five() {
        let cards = [
            Card::new(Rank::Six, Suit::Diamonds),
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Eight, Suit::Diamonds),
            Card::new(Rank::Nine, Suit::Diamonds),
            Card::new(Rank::Ten, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 5);
    }

    #[test]
    fn ace_low_sequence() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Spades),
            Card::new(Rank::Three, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
    }

    #[test]
    fn ace_high_sequence() {
        let cards = [
            Card::new(Rank::Queen, Suit::Clubs),
            Card::new(Rank::King, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Clubs),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
    }

    #[test]
    fn king_ace_two_no_wrap() {
        let cards = [
            Card::new(Rank::King, Suit::Diamonds),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Two, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn full_suit_sequence() {
        let cards: Vec<Card> = [
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
        ]
        .iter()
        .map(|&r| Card {
            rank: r,
            suit: Suit::Hearts,
        })
        .collect();
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 13);
    }

    // ── Jokers in Sets ──────────────────────────────────────────────────

    #[test]
    fn set_two_cards_one_joker() {
        let cards = [
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Ten, Suit::Diamonds),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn set_one_card_two_jokers() {
        let cards = [Card::new(Rank::Jack, Suit::Hearts)];
        let sol = solve(&cards, 2).unwrap();
        validate_solution(&sol, &cards, 2);
        assert_cards_placed(&sol, 1);
        assert_jokers_used(&sol, 2);
        assert_eq!(sol.groups().len(), 1);
    }

    // ── Jokers in Sequences ─────────────────────────────────────────────

    #[test]
    fn sequence_with_gap_filled_by_joker() {
        // 3♠ _♠ 5♠ — joker fills the 4
        let cards = [
            Card::new(Rank::Three, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
    }

    #[test]
    fn sequence_joker_extends_end() {
        // 7♦ 8♦ + joker as 6♦ or 9♦
        let cards = [
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Eight, Suit::Diamonds),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
    }

    #[test]
    fn sequence_two_jokers_one_card() {
        let cards = [Card::new(Rank::Six, Suit::Clubs)];
        let sol = solve(&cards, 2).unwrap();
        validate_solution(&sol, &cards, 2);
        assert_cards_placed(&sol, 1);
        assert_jokers_used(&sol, 2);
    }

    // ── Pure Joker Groups ───────────────────────────────────────────────

    #[test]
    fn three_jokers_form_group() {
        let sol = solve(&[], 3).unwrap();
        validate_solution(&sol, &[], 3);
        assert_jokers_used(&sol, 3);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn four_jokers_single_group() {
        let sol = solve(&[], 4).unwrap();
        validate_solution(&sol, &[], 4);
        assert_jokers_used(&sol, 4);
    }

    #[test]
    fn six_jokers_two_groups() {
        let sol = solve(&[], 6).unwrap();
        validate_solution(&sol, &[], 6);
        assert_jokers_used(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn five_jokers_all_used() {
        // 5 jokers can form a single sequence of length 5 (pure joker)
        let sol = solve(&[], 5).unwrap();
        validate_solution(&sol, &[], 5);
        assert_jokers_used(&sol, 5);
    }

    #[test]
    fn two_jokers_alone_cant_form_group() {
        let sol = solve(&[], 2).unwrap();
        validate_solution(&sol, &[], 2);
        assert_jokers_used(&sol, 0);
        assert_eq!(sol.groups().len(), 0);
    }

    #[test]
    fn one_joker_alone() {
        let sol = solve(&[], 1).unwrap();
        validate_solution(&sol, &[], 1);
        assert_jokers_used(&sol, 0);
    }

    // ── Multiple Groups ─────────────────────────────────────────────────

    #[test]
    fn two_disjoint_sets() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::King, Suit::Clubs),
            Card::new(Rank::King, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn set_and_sequence() {
        let cards = [
            Card::new(Rank::Three, Suit::Spades),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Hearts),
            Card::new(Rank::Nine, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn two_sequences_same_suit() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Spades),
            Card::new(Rank::Three, Suit::Spades),
            Card::new(Rank::Eight, Suit::Spades),
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    // ── Optimization / Card Allocation ──────────────────────────────────

    #[test]
    fn card_shared_between_set_and_sequence() {
        // 5♠ could join set (5♠,5♣,5♦) or sequence (4♠,5♠,6♠).
        // Either way only 3 cards get placed.
        let cards = [
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Five, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Six, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
    }

    #[test]
    fn prefer_more_cards_placed() {
        let cards = [
            // Set of 5s would use 3 cards
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Five, Suit::Diamonds),
            // Sequence 4♠-7♠ would use 4 (stealing 5♠ from set), breaking set
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Seven, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        // Sequence 4♠5♠6♠7♠ = 4 cards beats set of 3
        assert_cards_placed(&sol, 4);
    }

    #[test]
    fn maximize_with_joker_allocation() {
        let cards = [
            Card::new(Rank::Queen, Suit::Spades),
            Card::new(Rank::Queen, Suit::Diamonds),
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Four, Suit::Hearts),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        // Joker completes one partial group → 2 cards + 1 joker
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
    }

    #[test]
    fn joker_enables_two_groups() {
        let cards = [
            Card::new(Rank::Jack, Suit::Spades),
            Card::new(Rank::Jack, Suit::Clubs),
            Card::new(Rank::Ten, Suit::Diamonds),
            Card::new(Rank::Jack, Suit::Diamonds),
            Card::new(Rank::Queen, Suit::Diamonds),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        // Sequence T♦J♦Q♦ + set J♠J♣+joker → 5 cards placed
        assert_cards_placed(&sol, 5);
        assert_jokers_used(&sol, 1);
        assert_eq!(sol.groups().len(), 2);
    }

    // ── Duplicate Deck ──────────────────────────────────────────────────

    #[test]
    fn duplicate_cards_two_sets() {
        let cards = [
            Card::new(Rank::Eight, Suit::Spades),
            Card::new(Rank::Eight, Suit::Clubs),
            Card::new(Rank::Eight, Suit::Diamonds),
            Card::new(Rank::Eight, Suit::Spades),
            Card::new(Rank::Eight, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn duplicate_in_sequence() {
        let cards = [
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Six, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 6);
        assert_eq!(sol.groups().len(), 2);
    }

    // ── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn all_cards_one_suit_no_sequence() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn almost_a_set_only_two() {
        let cards = [
            Card::new(Rank::Two, Suit::Spades),
            Card::new(Rank::Two, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn joker_completes_ace_high_sequence() {
        let cards = [
            Card::new(Rank::Queen, Suit::Clubs),
            Card::new(Rank::King, Suit::Clubs),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
    }

    #[test]
    fn joker_completes_ace_low_sequence() {
        let cards = [
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Two, Suit::Diamonds),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 1);
    }

    #[test]
    fn many_jokers_with_cards() {
        // 5♥ _ _ 8♥ with 3 jokers → one sequence spanning 5-8 (2 jokers) or 5-9 (3 jokers)
        let cards = [
            Card::new(Rank::Five, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Hearts),
        ];
        let sol = solve(&cards, 3).unwrap();
        validate_solution(&sol, &cards, 3);
        assert_cards_placed(&sol, 2);
        assert_jokers_used(&sol, 3);
    }

    #[test]
    fn large_mixed_input() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Hearts),
            Card::new(Rank::Six, Suit::Hearts),
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::Two, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        // Aces set (3) + hearts sequence (4) = 7
        assert_cards_placed(&sol, 7);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn all_52_cards_in_sets() {
        let cards: Vec<Card> = [
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
        ]
        .iter()
        .flat_map(|&r| {
            [Suit::Spades, Suit::Clubs, Suit::Diamonds, Suit::Hearts]
                .iter()
                .map(move |&s| Card { rank: r, suit: s })
        })
        .collect();
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 52);
    }

    #[test]
    fn no_valid_groups_returns_empty() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Jack, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
        assert_eq!(sol.groups().len(), 0);
    }

    #[test]
    fn joker_not_wasted_when_set_already_complete() {
        // Perfect set of 3 + spare joker: joker may extend to 4 or stay unused
        let cards = [
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::Nine, Suit::Clubs),
            Card::new(Rank::Nine, Suit::Diamonds),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 3);
    }

    #[test]
    fn sequence_not_broken_by_wrong_suit() {
        let cards = [
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Six, Suit::Spades),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 0);
    }

    #[test]
    fn interlocking_groups() {
        // 3x3 grid: can be 3 sequences (by suit) or 3 sets (by rank)
        let cards = [
            Card::new(Rank::Three, Suit::Spades),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Four, Suit::Clubs),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Diamonds),
            Card::new(Rank::Five, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 9);
    }

    #[test]
    fn partial_placement_is_optimal() {
        let cards = [
            Card::new(Rank::Two, Suit::Spades),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Two, Suit::Diamonds),
            Card::new(Rank::Nine, Suit::Hearts),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
        assert_eq!(sol.groups().len(), 1);
    }

    #[test]
    fn seven_jokers() {
        let sol = solve(&[], 7).unwrap();
        validate_solution(&sol, &[], 7);
        assert_jokers_used(&sol, 7);
    }

    #[test]
    fn sequence_jqk() {
        let cards = [
            Card::new(Rank::Jack, Suit::Diamonds),
            Card::new(Rank::Queen, Suit::Diamonds),
            Card::new(Rank::King, Suit::Diamonds),
        ];
        let sol = solve(&cards, 0).unwrap();
        validate_solution(&sol, &cards, 0);
        assert_cards_placed(&sol, 3);
    }

    #[test]
    fn long_sequence_with_joker_gaps() {
        // 2♣ _ 4♣ 5♣ _ 7♣ — jokers fill 3♣ and 6♣
        let cards = [
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Four, Suit::Clubs),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Clubs),
        ];
        let sol = solve(&cards, 2).unwrap();
        validate_solution(&sol, &cards, 2);
        assert_cards_placed(&sol, 4);
        assert_jokers_used(&sol, 2);
    }

    #[test]
    fn set_plus_pure_joker_group() {
        let cards = [
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ];
        let sol = solve(&cards, 3).unwrap();
        validate_solution(&sol, &cards, 3);
        assert_cards_placed(&sol, 3);
        assert_jokers_used(&sol, 3);
        assert_eq!(sol.groups().len(), 2);
    }

    #[test]
    fn joker_used_in_best_allocation() {
        // Without joker: only hearts sequence (3 cards).
        // With joker completing spade/diamond set: 5 total.
        let cards = [
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Six, Suit::Diamonds),
            Card::new(Rank::Ten, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Hearts),
            Card::new(Rank::Queen, Suit::Hearts),
        ];
        let sol = solve(&cards, 1).unwrap();
        validate_solution(&sol, &cards, 1);
        assert_cards_placed(&sol, 5);
        assert_jokers_used(&sol, 1);
        assert_eq!(sol.groups().len(), 2);
    }
}
