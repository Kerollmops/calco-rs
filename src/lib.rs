extern crate rand;
extern crate roulette_wheel;

use std::iter::FromIterator;
use rand::{Rng, thread_rng, ThreadRng};
use roulette_wheel::RouletteWheel;

pub mod traits;
pub mod crossover;

use traits::{Evaluate, Mutate, Reproduce};

// Termination
// This generational process is repeated until a termination condition has been reached.
// Common terminating conditions are:

// - A solution is found that satisfies minimum criteria
// - Fixed number of generations reached
// - Allocated budget (computation time/money) reached
// - The highest ranking solution's fitness is reaching or has reached a plateau such
//       that successive iterations no longer produce better results
// - Manual inspection
// - Combinations of the above

/// https://en.wikipedia.org/wiki/Genetic_algorithm#Termination
#[derive(Debug)]
pub struct StopConditions {
    /// A solution is found that satisfies minimum criteria.
    pub reach_fitness: Option<f32>,

    /// Fixed number of generations reached.
    pub number_generations: Option<usize>,

    /// The highest ranking solution's fitness is reaching or
    /// has reached a plateau such that successive iterations no longer produce better results.
    pub stagnant_results: Option<f32>,
}

#[derive(Debug)]
pub struct Calco<R: Rng, T> {
    rng: R,
    population: Vec<T>
}

use std::fmt::Debug;

impl<R: Rng + Clone, T: Debug + Clone + Evaluate + Mutate + Reproduce> Iterator for Calco<R, T> {
    type Item = (f32, T);

    fn next(&mut self) -> Option<Self::Item> {
        let rw: RouletteWheel<_> = self.population.iter().cloned()
                                    .map(|ind| (ind.evaluate(), ind))
                                    .collect();

        let new_pop: Vec<_> = rw.into_iter().map(|(_, ind)| ind).collect();
        let mut rng = self.rng.clone();
        let iter = new_pop.chunks(2).flat_map(|parents| {
                        if parents.len() == 2 {
                            // println!("{:?} + {:?} = ❤️", parents[0], parents[1]);
                            let children = parents[0].reproduce(&parents[1], &mut rng);
                            children.into_iter()
                        }
                        else { parents.to_vec().into_iter() }
                    });
        self.population.clear();
        self.population.extend(iter);

        // TODO: really ugly
        let (fit, best) = self.population.iter()
                            .fold(None, |acc, ind| {
                                match acc {
                                    Some((bfit, best)) => {
                                        let fit = ind.evaluate();
                                        if fit > bfit {
                                            Some((fit, ind))
                                        } else {
                                            Some((bfit, best))
                                        }
                                    },
                                    None => Some((ind.evaluate(), ind)),
                                }
                            }).expect("Can't find best value");
        println!("pop len: {:?}", self.population.len());
        Some((fit, best.clone()))
    }
}

impl<T: Evaluate + Mutate + Reproduce> FromIterator<T> for Calco<ThreadRng, T> {
    fn from_iter<I>(iter: I) -> Calco<ThreadRng, T> where I: IntoIterator<Item=T> {
        Calco {
            rng: thread_rng(),
            population: iter.into_iter().collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, StdRng};
    use Calco;
    use traits::{Evaluate, Mutate, Reproduce};

    const SEED: [usize; 4] = [4, 2, 42, 4242];
    const global_minimum: (f32, f32) = (15.0, -15.0);

    #[derive(Debug, Copy, Clone)]
    struct SimpleIndividual {
        x: f32,
        y: f32
    }

    impl Evaluate for SimpleIndividual {
        fn evaluate(&self) -> f32 {
            // note: inverse euclidean distance
            let (gx, gy) = global_minimum;
            let x = self.x - gx;
            let y = self.y - gy;
            1.0 / (x * x + y * y).sqrt()
        }
    }

    impl Mutate for SimpleIndividual {
        fn mutate<R: Rng>(&mut self, rng: &mut R) {
            match rng.gen() { // ugly but... why not
                true => self.x = rng.gen(),
                false => self.y = rng.gen(),
            }
        }
    }

    impl Reproduce for SimpleIndividual {
        fn reproduce<'a, R: Rng>(&self, father: &Self, rng: &mut R) -> Vec<Self> {
            let mut children = Vec::with_capacity(2);
            match rng.gen() { // ugly but... why not
                true => {
                    children.push(SimpleIndividual { x: self.x, y: father.y });
                    children.push(SimpleIndividual { x: father.x, y: self.y });
                },
                false => {
                    children.push(SimpleIndividual {
                        x: (self.x + father.x) / 2.0,
                        y: (self.y + father.y) / 2.0
                    });
                    children.push(SimpleIndividual {
                        x: self.x + father.x,
                        y: self.y + father.y
                    });
                },
            }
            children
        }
    }

    #[test]
    fn iterator_simple() {
        let calco: Calco<_, _> = StdRng::from_seed(&SEED).gen_iter()
                                .take(100)
                                .map(|(x, y)| SimpleIndividual{ x: x, y: y })
                                .collect();

        for (gen, best) in calco.enumerate().take(10000) {
            println!("gen: {:?}, best: {:?}", gen, best);
        }
    }
}
