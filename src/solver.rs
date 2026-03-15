use std::collections::{HashMap, HashSet};

use good_lp::{
    Expression, ProblemVariables, Solution, SolverModel, Variable, constraint, default_solver,
    solvers::microlp::MicroLpProblem, variable,
};

use crate::types::{self, Card, Group, RANKS, Rank, SUITS, Suit};

const SET_MAX: usize = 4; // max cards in a set
const SEQ_MAX: usize = 14; // max cards in a sequence (A 2 3 … K A)

type CardMat = [[usize; SET_MAX + 1]; SEQ_MAX + 1]; // We have an extra row/column for jokers

/// Compute a matrix of card counts indexed by rank and suit.
/// Index 13 mirrors index 0 (Ace) for ace-high sequence detection.
fn card_matrix(group: &Group) -> CardMat {
    let mut mat: CardMat = [[0; SET_MAX + 1]; SEQ_MAX + 1];
    for card in &group.cards {
        mat[card.rank.index()][card.suit.index()] += 1;
    }
    mat[13] = mat[0];
    mat
}

/// Compute all possible groups we can obtain by substituting jokers into an existing group.
/// For each non-empty subset of non-joker cards to remove (up to jokers_left, keeping at least
/// 1 non-joker card), produce a variant where those cards are replaced by joker cards.
fn joker_substitute(group: Group, jokers_left: usize) -> Vec<Group> {
    let mut non_joker_indices = Vec::new();
    for (i, &card) in group.cards.iter().enumerate() {
        if card != Card::JOKER {
            non_joker_indices.push(i);
        }
    }

    let n = non_joker_indices.len();
    if n == 0 {
        return vec![group];
    }

    let max_remove = jokers_left.min(n - 1); // keep at least 1 non-joker
    let mut expanded = Vec::new();
    for mask in 0u64..(1u64 << n) {
        let remove_count = mask.count_ones() as usize;
        if remove_count > max_remove {
            continue;
        }

        let mut cards = group.cards.clone();
        for bit in 0..n {
            if mask & (1u64 << bit) != 0 {
                cards[non_joker_indices[bit]] = Card::JOKER;
            }
        }
        expanded.push(Group { cards });
    }
    expanded
}

/// Enumerate all valid sequences (3-14 consecutive ranks of a single suit, or jokers).
fn enumerate_sequences(mat: &CardMat) -> Vec<Group> {
    let mut groups = Vec::new();
    let jokers = mat[Rank::Joker.index()][Suit::Joker.index()];

    for suit in SUITS {
        for start in 0..(SEQ_MAX - 2) {
            let mut jokers_used = 0;
            for end in start..SEQ_MAX {
                // Greedily extend the sequence and fill holes with jokers
                if mat[end][suit.index()] == 0 {
                    jokers_used += 1;
                    if jokers_used > jokers {
                        break;
                    }
                }

                let len = end - start + 1;
                if len >= 3 {
                    let mut cards: Vec<Card> = Vec::with_capacity(len);
                    for i in start..=end {
                        if mat[i][suit.index()] > 0 {
                            cards.push(Card::new(Rank::from_index(i), suit));
                        } else {
                            cards.push(Card::JOKER);
                        }
                    }
                    let group = Group { cards };
                    groups.extend(joker_substitute(group, jokers - jokers_used));
                }
            }
        }
    }

    // Pure joker sequences
    for j in 3..=jokers.min(SEQ_MAX) {
        groups.push(Group {
            cards: vec![Card::JOKER; j],
        });
    }

    groups
}

/// Enumerate all valid sets (3–4 cards of the same rank, each a different suit, or jokers).
/// We don't generate sets with <=1 non-joker card, as those could also be sequences which
/// are potentially longer.
fn enumerate_sets(mat: &CardMat) -> Vec<Group> {
    let mut groups = Vec::new();
    let jokers = mat[Rank::Joker.index()][Suit::Joker.index()];

    for rank in RANKS {
        let suits_present: Vec<Suit> = SUITS
            .iter()
            .copied()
            .filter(|&s| mat[rank.index()][s.index()] > 0)
            .collect();

        let n = suits_present.len();
        if n < 2 {
            continue;
        }

        // Choose any subset of the available suits to keep as real cards.
        for mask in 0usize..(1usize << n) {
            let real_count = mask.count_ones() as usize;
            if real_count < 2 {
                continue;
            }

            let chosen_suits: Vec<Suit> = suits_present
                .iter()
                .enumerate()
                .filter_map(|(bit, &s)| ((mask & (1 << bit)) != 0).then_some(s))
                .collect();

            // Pad with jokers to reach a legal set size (3 or 4).
            for total_size in real_count.max(3)..=SET_MAX {
                let jokers_needed = total_size - real_count;
                if jokers_needed > jokers {
                    continue;
                }

                let mut cards: Vec<Card> =
                    chosen_suits.iter().map(|&s| Card::new(rank, s)).collect();

                cards.extend(std::iter::repeat_n(Card::JOKER, jokers_needed));
                groups.push(Group { cards });
            }
        }
    }

    groups
}

/// Deduplicate groups that differ only in joker placement.
fn dedup_groups(groups: Vec<Group>) -> Vec<Group> {
    let mut seen = HashSet::new();
    groups
        .into_iter()
        .filter(|g| {
            let mut key = g.cards.clone();
            key.sort();
            seen.insert(key)
        })
        .collect()
}

/// Compute the maximum number of times each group can be selected based on the input card counts.
fn group_max(full_mat: &CardMat, groups: &[Group]) -> Vec<usize> {
    groups
        .iter()
        .map(|group| {
            let mut card_counts: HashMap<Card, usize> = HashMap::new();
            for &c in &group.cards {
                *card_counts.entry(c).or_default() += 1;
            }
            card_counts
                .iter()
                .map(|(c, &need)| full_mat[c.rank.index()][c.suit.index()] / need)
                .min()
                .unwrap_or(0)
        })
        .collect()
}

/// Compute a mapping from each card to the indices of groups that contain it.
/// Note that a card may be contained multiple times in the same group.
fn card_to_groups(groups: &[Group]) -> HashMap<Card, Vec<usize>> {
    let mut card_to_groups: HashMap<Card, Vec<usize>> = HashMap::new();
    for (idx, group) in groups.iter().enumerate() {
        for &card in &group.cards {
            card_to_groups.entry(card).or_default().push(idx);
        }
    }
    card_to_groups
}

/// Build the Mixed-Integer Linear Program to solve the card grouping problem.
fn build_lp(
    full_mat: &CardMat,
    board: &Group,
    hand: &Group,
    groups: &[Group],
) -> (MicroLpProblem, Vec<Variable>) {
    let group_max = group_max(full_mat, groups);
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

    // Constraint: each input card must be used exactly as many times as it appears in the input
    //  - each board card must be used exactly once
    //  - each hand card can be used at most once
    let card_set: HashSet<Card> = board
        .cards
        .iter()
        .chain(hand.cards.iter())
        .cloned()
        .collect();
    let board_mat = card_matrix(board);
    let hand_mat = card_matrix(hand);
    for card in card_set.iter() {
        let lhs: Expression = card_to_groups
            .get(card)
            .map(|group_indices| group_indices.iter().map(|&idx| x[idx]).sum())
            .unwrap_or_else(|| 0.0.into());
        let board_bound = board_mat[card.rank.index()][card.suit.index()] as f64;
        let hand_bound = hand_mat[card.rank.index()][card.suit.index()] as f64;
        model = model.with(constraint!(lhs.clone() >= board_bound));
        model = model.with(constraint!(lhs <= hand_bound + board_bound));
    }

    (model, x)
}

pub fn solve(board: &Group, hand: &Group) -> Option<types::Solution> {
    let all_cards = board + hand;
    let full_mat = card_matrix(&all_cards);

    let groups = dedup_groups([enumerate_sequences(&full_mat), enumerate_sets(&full_mat)].concat());
    if groups.is_empty() && hand.cards.is_empty() {
        return None;
    }

    let (model, x) = build_lp(&full_mat, board, hand, &groups);
    let solution = model.solve().ok()?;

    let mut result = types::Solution::default();
    let mut used_cards = Group::default();
    for (idx, group) in groups.into_iter().enumerate() {
        let val = solution.value(x[idx]).round() as usize;
        for _ in 0..val {
            result.add_group(group.clone());
            used_cards += group.clone();
        }
    }

    let mut used_mat = card_matrix(&used_cards);
    for &card in &hand.cards {
        let (r, s) = (card.rank.index(), card.suit.index());
        if used_mat[r][s] < full_mat[r][s] {
            result.add_remaining(card);
            used_mat[r][s] += 1;
        }
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Card, Group, Rank, Suit};

    fn is_valid_set(g: &Group) -> bool {
        let len = g.cards.len();
        if !(3..=SET_MAX).contains(&len) {
            return false;
        }

        let mut rank: Option<Rank> = None;
        let mut suits = HashSet::new();

        for &card in &g.cards {
            if card == Card::JOKER {
                continue;
            }

            match rank {
                Some(r) if r != card.rank => return false,
                None => rank = Some(card.rank),
                _ => {}
            }

            if !suits.insert(card.suit) {
                return false;
            }
        }

        true
    }

    fn is_valid_sequence(g: &Group) -> bool {
        let len = g.cards.len();
        if !(3..=SEQ_MAX).contains(&len) {
            return false;
        }

        let mut suit = Suit::Joker;
        let mut rank_idx = 0;
        for (i, &card) in g.cards.iter().enumerate() {
            if card == Card::JOKER {
                continue;
            }

            suit = card.suit;
            if card.rank == Rank::Ace && i > 0 {
                rank_idx = Rank::ACE_HIGH.saturating_sub(i);
            } else {
                rank_idx = card.rank.index().saturating_sub(i);
            }
            break;
        }

        for &card in &g.cards {
            if card != Card::JOKER
                && (rank_idx > Rank::ACE_HIGH
                    || card.suit != suit
                    || card.rank.index() != Rank::from_index(rank_idx).index())
            {
                return false;
            }

            rank_idx += 1;
        }

        true
    }

    fn is_valid_group(g: &Group) -> bool {
        if g.cards.is_empty() {
            return false;
        }
        is_valid_set(g) || is_valid_sequence(g)
    }

    fn g(cards: impl Into<Vec<Card>>) -> Group {
        Group {
            cards: cards.into(),
        }
    }

    #[track_caller]
    fn assert_solution(
        board: Group,
        hand: Group,
        expected_groups: Option<usize>,
        expected_remaining: Option<usize>,
    ) {
        let input_str = format!("Board: {board}\nHand: {hand}");
        let sol = solve(&board, &hand).unwrap_or_else(|| {
            panic!("Expected exact cover, but solve() returned None. \n{input_str}")
        });

        for (i, g) in sol.groups().iter().enumerate() {
            assert!(
                is_valid_group(g),
                "Group {i} is not a valid set or sequence: {g}"
            );
        }

        if let Some(n) = expected_groups {
            assert_eq!(
                sol.groups().len(),
                n,
                "Expected {n} groups, got {}.\n{input_str}\nSolution:\n{sol}",
                sol.groups().len()
            );
        }

        if let Some(n) = expected_remaining {
            assert_eq!(
                sol.remaining().cards.len(),
                n,
                "Expected {n} remaining cards, got {}.\n{input_str}\nSolution:\n{sol}",
                sol.remaining().cards.len()
            );
        }

        let used_cards = sol
            .groups()
            .iter()
            .fold(sol.remaining().clone(), |acc, g| &acc + g);
        let input_cards = &board + &hand;
        assert_eq!(
            card_matrix(&used_cards),
            card_matrix(&input_cards),
            "Solution does not use exactly the input cards.\n{input_str}\nSolution:\n{sol}"
        );
    }

    #[track_caller]
    fn assert_exact_solution(board: Group, expected_groups: Option<usize>) {
        assert_solution(board, Group::default(), expected_groups, Some(0));
    }

    #[track_caller]
    fn assert_no_solution(board: Group, hand: Group) {
        let sol = solve(&board, &hand);
        assert!(
            sol.is_none(),
            "Expected no exact cover, but got a solution.\nBoard: {board}\nHand: {hand}\nSolution:\n{}",
            sol.unwrap()
        );
    }

    #[track_caller]
    fn assert_no_exact_solution(board: Group) {
        assert_no_solution(board, Group::default());
    }

    // ── Trivial impossibility / exact-cover semantics ───────────────────

    #[test]
    fn too_short_inputs_are_unsolved() {
        assert_no_exact_solution(g([]));
        assert_no_exact_solution(g([Card::new(Rank::Ace, Suit::Spades)]));
        assert_no_exact_solution(g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Spades),
        ]));
        assert_no_exact_solution(g([Card::JOKER; 2]));
    }

    #[test]
    fn unrelated_cards_have_no_solution() {
        assert_no_exact_solution(g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Jack, Suit::Hearts),
        ]));
    }

    #[test]
    fn leftover_card_rejects_partial_cover() {
        assert_no_exact_solution(g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Seven, Suit::Hearts),
        ]));
    }

    // ── Sets ────────────────────────────────────────────────────────────

    #[test]
    fn set_of_three_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::Five, Suit::Clubs),
                Card::new(Rank::Five, Suit::Diamonds),
            ]),
            Some(1),
        );
    }

    #[test]
    fn set_of_four_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::King, Suit::Spades),
                Card::new(Rank::King, Suit::Clubs),
                Card::new(Rank::King, Suit::Diamonds),
                Card::new(Rank::King, Suit::Hearts),
            ]),
            Some(1),
        );
    }

    #[test]
    fn duplicate_suit_does_not_make_a_set() {
        assert_no_exact_solution(g([
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
        ]));
    }

    #[test]
    fn duplicate_deck_can_make_two_sets() {
        assert_exact_solution(
            g([
                Card::new(Rank::Eight, Suit::Spades),
                Card::new(Rank::Eight, Suit::Clubs),
                Card::new(Rank::Eight, Suit::Diamonds),
                Card::new(Rank::Eight, Suit::Hearts),
                Card::new(Rank::Eight, Suit::Spades),
                Card::new(Rank::Eight, Suit::Clubs),
                Card::new(Rank::Eight, Suit::Diamonds),
                Card::new(Rank::Eight, Suit::Hearts),
            ]),
            Some(2),
        );
    }

    // ── Sequences ───────────────────────────────────────────────────────

    #[test]
    fn simple_sequence_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Three, Suit::Hearts),
                Card::new(Rank::Four, Suit::Hearts),
                Card::new(Rank::Five, Suit::Hearts),
            ]),
            Some(1),
        );
    }

    #[test]
    fn ace_low_sequence_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::Two, Suit::Spades),
                Card::new(Rank::Three, Suit::Spades),
            ]),
            Some(1),
        );
    }

    #[test]
    fn ace_high_sequence_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Queen, Suit::Clubs),
                Card::new(Rank::King, Suit::Clubs),
                Card::new(Rank::Ace, Suit::Clubs),
            ]),
            Some(1),
        );
    }

    #[test]
    fn maximal_14_card_sequence_is_solved() {
        let mut cards: Vec<Card> = RANKS.iter().map(|&r| Card::new(r, Suit::Hearts)).collect();
        cards.push(Card::new(Rank::Ace, Suit::Hearts)); // Ace-high copy from duplicate deck
        assert_exact_solution(Group { cards }, Some(1));
    }

    #[test]
    fn sequence_cannot_wrap_around() {
        assert_no_exact_solution(g([
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::Two, Suit::Hearts),
        ]));
    }

    #[test]
    fn wrong_suit_breaks_sequence() {
        assert_no_exact_solution(g([
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Six, Suit::Spades),
        ]));
    }

    // ── Jokers ──────────────────────────────────────────────────────────

    #[test]
    fn joker_fills_internal_gaps_in_sequence() {
        assert_exact_solution(
            g([
                Card::new(Rank::Two, Suit::Clubs),
                Card::JOKER,
                Card::new(Rank::Four, Suit::Clubs),
                Card::new(Rank::Five, Suit::Clubs),
                Card::JOKER,
                Card::new(Rank::Seven, Suit::Clubs),
            ]),
            Some(1),
        );
    }

    #[test]
    fn joker_can_complete_ace_edge_sequences() {
        assert_exact_solution(
            g([
                Card::new(Rank::Queen, Suit::Clubs),
                Card::new(Rank::King, Suit::Clubs),
                Card::JOKER,
            ]),
            Some(1),
        );

        assert_exact_solution(
            g([
                Card::new(Rank::Ace, Suit::Diamonds),
                Card::new(Rank::Two, Suit::Diamonds),
                Card::JOKER,
            ]),
            Some(1),
        );
    }

    #[test]
    fn one_real_card_and_two_jokers_is_still_solvable() {
        assert_exact_solution(
            g([Card::new(Rank::Six, Suit::Clubs), Card::JOKER, Card::JOKER]),
            Some(1),
        );
    }

    #[test]
    fn two_real_cards_and_a_joker_can_form_a_set() {
        assert_exact_solution(
            g([
                Card::new(Rank::Ten, Suit::Spades),
                Card::new(Rank::Ten, Suit::Diamonds),
                Card::JOKER,
            ]),
            Some(1),
        );
    }

    #[test]
    fn pure_jokers_can_form_one_or_multiple_groups() {
        assert_exact_solution(g([Card::JOKER; 3]), Some(1));
        assert_exact_solution(g([Card::JOKER; 15]), Some(2));
    }

    #[test]
    fn joker_cannot_enable_wraparound_sequence() {
        assert_no_exact_solution(g([
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Hearts),
            Card::JOKER,
            Card::new(Rank::Two, Suit::Hearts),
        ]));
    }

    // ── Multiple groups / ambiguous allocation ─────────────────────────

    #[test]
    fn two_disjoint_sets_are_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::Ace, Suit::Clubs),
                Card::new(Rank::Ace, Suit::Diamonds),
                Card::new(Rank::King, Suit::Spades),
                Card::new(Rank::King, Suit::Clubs),
                Card::new(Rank::King, Suit::Hearts),
            ]),
            Some(2),
        );
    }

    #[test]
    fn set_and_sequence_are_solved_together() {
        assert_exact_solution(
            g([
                Card::new(Rank::Three, Suit::Spades),
                Card::new(Rank::Three, Suit::Clubs),
                Card::new(Rank::Three, Suit::Diamonds),
                Card::new(Rank::Seven, Suit::Hearts),
                Card::new(Rank::Eight, Suit::Hearts),
                Card::new(Rank::Nine, Suit::Hearts),
            ]),
            Some(2),
        );
    }

    #[test]
    fn duplicate_deck_can_make_two_identical_sequences() {
        assert_exact_solution(
            g([
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::Six, Suit::Spades),
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::Six, Suit::Spades),
            ]),
            Some(2),
        );
    }

    #[test]
    fn shared_card_requires_correct_allocation() {
        assert_exact_solution(
            g([
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::Five, Suit::Clubs),
                Card::new(Rank::Five, Suit::Diamonds),
                Card::new(Rank::Five, Suit::Hearts),
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Six, Suit::Spades),
            ]),
            Some(2),
        );
    }

    #[test]
    fn joker_can_enable_two_groups() {
        assert_exact_solution(
            g([
                Card::new(Rank::Jack, Suit::Spades),
                Card::new(Rank::Jack, Suit::Clubs),
                Card::new(Rank::Jack, Suit::Diamonds),
                Card::new(Rank::Ten, Suit::Diamonds),
                Card::JOKER,
                Card::new(Rank::Queen, Suit::Diamonds),
            ]),
            Some(2),
        );
    }

    #[test]
    fn set_plus_pure_joker_group_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Ace, Suit::Spades),
                Card::new(Rank::Ace, Suit::Clubs),
                Card::new(Rank::Ace, Suit::Diamonds),
                Card::JOKER,
                Card::JOKER,
                Card::JOKER,
            ]),
            Some(2),
        );
    }

    #[test]
    fn interlocking_exact_cover_is_solved() {
        assert_exact_solution(
            g([
                Card::new(Rank::Three, Suit::Spades),
                Card::new(Rank::Four, Suit::Spades),
                Card::new(Rank::Five, Suit::Spades),
                Card::new(Rank::Three, Suit::Clubs),
                Card::new(Rank::Four, Suit::Clubs),
                Card::new(Rank::Five, Suit::Clubs),
                Card::new(Rank::Three, Suit::Diamonds),
                Card::new(Rank::Four, Suit::Diamonds),
                Card::new(Rank::Five, Suit::Diamonds),
            ]),
            Some(3),
        );
    }

    #[test]
    fn all_52_cards_have_an_exact_cover() {
        let cards: Vec<Card> = RANKS
            .iter()
            .flat_map(|&r| SUITS.iter().map(move |&s| Card::new(r, s)))
            .collect();
        assert_exact_solution(g(cards), Some(4));
    }

    // ── Board / hand differentiation ────────────────────────────────────

    #[test]
    fn real_game_47_cards() {
        // This is the game that prompted making this solver.
        let board = g([
            Card::new(Rank::Queen, Suit::Spades),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Six, Suit::Hearts),
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Hearts),
            Card::new(Rank::Nine, Suit::Hearts),
            Card::new(Rank::Four, Suit::Diamonds),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Two, Suit::Diamonds),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Two, Suit::Hearts),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Eight, Suit::Spades),
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Ten, Suit::Hearts),
            Card::new(Rank::Ten, Suit::Diamonds),
            Card::new(Rank::Ten, Suit::Clubs),
            Card::new(Rank::Queen, Suit::Diamonds),
            Card::new(Rank::King, Suit::Diamonds),
            Card::new(Rank::Ace, Suit::Diamonds),
            Card::new(Rank::Six, Suit::Hearts),
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Six, Suit::Clubs),
            Card::new(Rank::Nine, Suit::Clubs),
            Card::new(Rank::Ten, Suit::Clubs),
            Card::JOKER,
            Card::new(Rank::Two, Suit::Diamonds),
            Card::new(Rank::Two, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Ten, Suit::Hearts),
            Card::new(Rank::Ten, Suit::Diamonds),
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Five, Suit::Diamonds),
            Card::new(Rank::Six, Suit::Diamonds),
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Eight, Suit::Diamonds),
        ]);
        let hand = g([
            Card::new(Rank::Queen, Suit::Clubs),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::Queen, Suit::Diamonds),
        ]);
        let full = &board + &hand;
        assert_solution(board, hand, None, Some(2));
        assert_no_exact_solution(full);
    }

    #[test]
    fn unplayable_hand_cards_remain() {
        let board = g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ]);
        let hand = g([
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Diamonds),
        ]);
        assert_solution(board, hand, Some(1), Some(2));
    }

    #[test]
    fn hand_completes_impossible_board() {
        let board = g([
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Four, Suit::Hearts),
        ]);
        let hand = g([Card::new(Rank::Five, Suit::Hearts)]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn hand_cannot_complete_impossible_board() {
        let board = g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Seven, Suit::Hearts),
        ]);
        let hand = g([Card::new(Rank::Jack, Suit::Diamonds)]);
        assert_no_solution(board, hand);
    }

    #[test]
    fn hand_extends_sequence() {
        let board = g([
            Card::new(Rank::Six, Suit::Spades),
            Card::new(Rank::Seven, Suit::Spades),
            Card::new(Rank::Eight, Suit::Spades),
        ]);
        let hand = g([Card::new(Rank::Nine, Suit::Spades)]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn hand_extends_set() {
        let board = g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Diamonds),
        ]);
        let hand = g([Card::new(Rank::Ace, Suit::Hearts)]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn partial_hand_usage() {
        let board = g([
            Card::new(Rank::Five, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Five, Suit::Diamonds),
        ]);
        let hand = g([
            Card::new(Rank::Five, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Clubs),
        ]);
        assert_solution(board, hand, Some(1), Some(1));
    }

    #[test]
    fn hand_card_enables_board_rearrangement() {
        let board = g([
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Hearts),
            Card::new(Rank::Six, Suit::Hearts),
            Card::new(Rank::Six, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Clubs),
        ]);
        let hand = g([Card::new(Rank::Eight, Suit::Clubs)]);
        assert_solution(board, hand, Some(2), Some(0));
    }

    #[test]
    fn empty_board_returns_hand() {
        let board = Group::default();
        let hand = g([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Seven, Suit::Hearts),
        ]);
        assert_solution(board, hand, Some(0), Some(2));
    }

    #[test]
    fn empty_board_hand_forms_set() {
        let board = Group::default();
        let hand = g([
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Ten, Suit::Clubs),
            Card::new(Rank::Ten, Suit::Diamonds),
        ]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn hand_joker_completes_board_sequence() {
        let board = g([
            Card::new(Rank::Nine, Suit::Diamonds),
            Card::new(Rank::Jack, Suit::Diamonds),
        ]);
        let hand = g([Card::JOKER]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn all_hand_cards_used_when_possible() {
        let board = g([
            Card::new(Rank::Three, Suit::Spades),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Spades),
        ]);
        let hand = g([
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Three, Suit::Diamonds),
            Card::new(Rank::Three, Suit::Hearts),
        ]);
        assert_solution(board, hand, Some(2), Some(0));
    }

    #[test]
    fn duplicate_hand_cards_partially_used() {
        let board = g([
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Two, Suit::Diamonds),
        ]);
        let hand = g([
            Card::new(Rank::Two, Suit::Hearts),
            Card::new(Rank::Two, Suit::Hearts),
        ]);
        assert_solution(board, hand, Some(1), Some(1));
    }

    #[test]
    fn hand_joins_sequences() {
        let board = g([
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Four, Suit::Clubs),
            Card::new(Rank::Eight, Suit::Clubs),
            Card::new(Rank::Nine, Suit::Clubs),
            Card::new(Rank::Ten, Suit::Clubs),
        ]);
        let hand = g([
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Six, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Clubs),
        ]);
        assert_solution(board, hand, Some(1), Some(0));
    }

    #[test]
    fn prefer_more_cards_over_fewer_groups() {
        let board = g([
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Four, Suit::Clubs),
            Card::new(Rank::Six, Suit::Clubs),
            Card::new(Rank::Seven, Suit::Clubs),
            Card::new(Rank::Eight, Suit::Clubs),
        ]);
        let hand = g([
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Six, Suit::Clubs),
        ]);
        assert_solution(board, hand, Some(2), Some(0));
    }
}
