use rand::Rng;

/// https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Single-point_crossover
pub fn single_point_crossover<T: Clone, R: Rng>(rng: &mut R, mother: &[T], father: &[T]) -> [Vec<T>; 2] {
    assert_eq!(mother.len(), father.len());
    let len = mother.len();

    // TODO: use https://doc.rust-lang.org/rand/rand/distributions/range/struct.Range.html
    let section = rng.gen_range(0, len);

    let (first_mother_split, second_mother_split) = mother.split_at(section);
    let (first_father_split, second_father_split) = father.split_at(section);

    let first_child: Vec<_> = first_mother_split.iter().chain(second_father_split).cloned().collect();
    let second_child: Vec<_> = first_father_split.iter().chain(second_mother_split).cloned().collect();

    assert_eq!(first_child.len(), second_child.len());
    [first_child, second_child]
}

/// https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_crossover
pub fn two_point_crossover<T: Clone, R: Rng>(rng: &mut R, mother: &[T], father: &[T]) -> [Vec<T>; 2] {
    assert_eq!(mother.len(), father.len());
    let len = mother.len();

    // TODO: use https://doc.rust-lang.org/rand/rand/distributions/range/struct.Range.html
    let first_section = rng.gen_range(0, len);
    let second_section = rng.gen_range(first_section, len);

    let (first_mother_split, second_mother_split, third_mother_split) = {
        let (first, second) = mother.split_at(first_section);
        let (second, third) = second.split_at(second_section);
        (first, second, third)
    };
    let (first_father_split, second_father_split, third_father_split) = {
        let (first, second) = father.split_at(first_section);
        let (second, third) = second.split_at(second_section);
        (first, second, third)
    };

    let first_child: Vec<_> = first_mother_split.iter()
                                .chain(second_father_split)
                                .chain(third_mother_split)
                                .cloned().collect();
    let second_child: Vec<_> = first_father_split.iter()
                                .chain(second_mother_split)
                                .chain(third_father_split)
                                .cloned().collect();

    assert_eq!(first_child.len(), second_child.len());
    [first_child, second_child]
}
