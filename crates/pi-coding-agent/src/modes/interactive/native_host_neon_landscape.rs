//! Terminal-cell artwork: a small striped disk behind an opaque mountain range.

pub(super) const WIDTH: usize = 46;
pub(super) const HEIGHT: usize = 9;

const SUN_LEFT: usize = 16;
const SUN: [&str; HEIGHT] = [
    "    ------    ",
    "              ",
    " ------------ ",
    "              ",
    "--------------",
    "              ",
    " ------------ ",
    "              ",
    "    ------    ",
];

const MOUNTAINS: [&str; HEIGHT] = [
    r"                                              ",
    r"                                 /\           ",
    r"       /\                       /  \          ",
    r"      /  \                     /    \         ",
    r"     /    \   /\              /      \        ",
    r"    /      \ /  \            /        \ /\    ",
    r"   /        /    \  /\      /          /  \   ",
    r"  /               \/  \    /               \  ",
    r"_/                     \__/                 \_",
];

pub(super) fn render() -> [String; HEIGHT] {
    let mut occluded = [false; WIDTH];
    std::array::from_fn(|y| {
        (0..WIDTH)
            .map(|x| {
                let ridge = MOUNTAINS[y].as_bytes()[x];
                if ridge != b' ' {
                    occluded[x] = true;
                    char::from(ridge)
                } else if occluded[x] {
                    // The terrain hides the disk below the ridge as well as on it.
                    ' '
                } else {
                    x.checked_sub(SUN_LEFT)
                        .and_then(|x| SUN[y].as_bytes().get(x))
                        .copied()
                        .map(char::from)
                        .unwrap_or(' ')
                }
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_round_disk_has_separated_symmetric_stripes() {
        assert_eq!(SUN.len(), 9);
        assert!(SUN.iter().all(|row| row.len() == 14));
        let widths: Vec<_> = SUN.iter().map(|row| row.trim().len()).collect();
        assert_eq!(widths, [6, 0, 12, 0, 14, 0, 12, 0, 6]);
    }

    #[test]
    fn continuous_foreground_hides_the_lower_disk_and_keeps_both_sides() {
        let rows = render();
        assert!(rows.iter().all(|row| row.is_ascii() && row.len() == WIDTH));
        let mut previous = None;
        for x in 0..WIDTH {
            let ridge: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|(_, row)| matches!(row.as_bytes()[x], b'/' | b'\\' | b'_'))
                .map(|(y, _)| y)
                .collect();
            assert_eq!(ridge.len(), 1, "one continuous ridge at column {x}");
            let y = ridge[0];
            if let Some(prev) = previous {
                assert!(y.abs_diff(prev) <= 1, "disconnected ridge at column {x}");
            }
            previous = Some(y);
            assert!(rows[y + 1..].iter().all(|row| row.as_bytes()[x] == b' '));
        }
        assert!(rows[2].contains("/\\")); // Peak to the left of the sun.
        assert!(rows[1].contains("/\\")); // Peak to the right of the sun.
        assert_eq!(&rows[4][16..30], "--------------");
        assert_eq!(&rows[6][17..29], "\\--/\\------/");
        assert!(
            !rows[8].contains('-'),
            "the bottom stripe is behind the terrain"
        );
    }
}
