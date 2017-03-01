extern crate rand;
extern crate roulette_wheel;

use rand::{Rng, thread_rng, ThreadRng};
use roulette_wheel::{RouletteWheel, IntoSelectIter};

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

impl<R: Rng, T> Calco<R, T> {
    pub fn new<I: Iterator<Item=T>>(population: I) -> Calco<ThreadRng, T> {
        Calco {
            rng: thread_rng(),
            population: population.collect()
        }
    }

    pub fn with_rng<I: Iterator<Item=T>>(rng: R, population: I) -> Calco<R, T> {
        Calco {
            rng: rng,
            population: population.collect()
        }
    }
}

// TODO: remove
use std::fmt::Debug;

impl<R, T> Iterator for Calco<R, T> where R: Rng + Clone, T: Debug + Clone + Evaluate + Mutate + Reproduce {
    type Item = (f32, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.population.sort_by(|a, b| b.evaluate().partial_cmp(&a.evaluate()).unwrap());
        let rw: RouletteWheel<_, _> = self.population.iter().cloned()
                                            .map(|ind| (ind.evaluate() as f64, ind))
                                            .collect();

        let new_pop: Vec<_> = IntoSelectIter::with_rng(self.rng.clone(), rw)
                                .into_iter()
                                .map(|(_, ind)| ind)
                                .collect();

        let iter: Vec<_> = new_pop.chunks(2).flat_map(|parents| {
                                if parents.len() == 2 {
                                    let children = parents[0].reproduce(&parents[1], &mut self.rng);
                                    children.into_iter()
                                }
                                else { parents.to_vec().into_iter() }
                            }).take(97).collect();

        self.population.truncate(3); // keep bests
        self.population.extend(iter);

        if self.rng.gen::<f32>() < 0.2 {
            if let Some(ind) = self.rng.choose_mut(self.population.as_mut_slice()) {
                ind.mutate(&mut self.rng);
            }
        }

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
        Some((fit, best.clone()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
