#![feature(slice_patterns)]

extern crate rand;
extern crate num;
extern crate rayon;
extern crate roulette_wheel;

use std::iter::{IntoIterator, Sum};
use std::cmp::PartialOrd;
use num::{Num, Zero, ToPrimitive, FromPrimitive};
use rand::{Rng, thread_rng, ThreadRng};
use rand::distributions::range::SampleRange;
use rayon::prelude::*;

pub mod traits;
pub mod crossover;

use traits::Individual;

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
pub struct Parameters {
    pub mutation_rate: f32,
    pub population_limit: Option<usize>,
    pub elite_threshold: f32,
    pub deletion_threshold: f32,
}

/// Quick and easy who-mates-and-who-dies algorithm with two thresholds.
/// The diagram below shows two typical threshold values which can be changed.
///
/// +-------------------------------------------+  1.0 Best Fitness
/// | Always a parent; lives another generation |
/// +-------------------------------------------+ ~0.9 Elite Threshold
/// |                                           |
/// | May be a parent; lives another generation |
/// |                                           |
/// +-------------------------------------------+ ~0.5 Fitness Deletion Threashold
/// |                                           |
/// | Does not survive this generation,         |
/// | Replaced by new offspring                 |
/// |                                           |
/// +-------------------------------------------+  0.0 Worst Fitness
#[derive(Debug)]
pub struct Calco<F, T, R> {
    rng: R,
    parameters: Parameters,
    population: Vec<(F, T)>
}

impl<F: Num + Zero, T: Individual<F>, R: Rng> Calco<F, T, R> {
    pub fn new<I: IntoIterator<Item=T>>(parameters: Parameters, population: I) -> Calco<F, T, ThreadRng> {
        Calco::with_rng(thread_rng(), parameters, population)
    }

    pub fn with_rng<I: IntoIterator<Item=T>>(rng: R, parameters: Parameters, population: I) -> Calco<F, T, R> {
        Calco {
            rng: rng,
            parameters: parameters,
            population: population.into_iter().map(|i| (F::zero(), i)).collect()
        }
    }
}

impl<F, T, R> Iterator for Calco<F, T, R>
    where F: Copy + Num + PartialOrd + SampleRange + Sum + ToPrimitive + FromPrimitive,
          R: Clone + Rng,
          T: Clone + Individual<F> {

    type Item = (F, T);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: do this elsewhere (specific trait ?)
        {
            for &mut (ref mut f, ref i) in self.population.iter_mut() {
                *f = i.evaluate();
            }
            // self.population.par_iter_mut()
            //                .for_each(|(ref mut f, ref i)| *f = i.evaluate());

            self.population.sort_by(|&(a, _), &(b, _)| b.partial_cmp(&a).unwrap());

            let len = self.population.len() as f32;
            let deletion_index = (self.parameters.deletion_threshold * len) as usize;
            let elite_index = (self.parameters.elite_threshold * len) as usize - deletion_index;
            let (winners, mut loosers) = self.population.split_at_mut(deletion_index);

            for &mut (ref mut fit, ref mut ind) in loosers.iter_mut() {
                // TODO: +1, -1 ?
                let &(_, ref mother) = self.rng.choose(&winners[..elite_index]).expect("Can't choose elite mother!");
                let &(_, ref father) = self.rng.choose(&winners).expect("Can't choose winner father!");

                *ind = mother.reproduce(&father, self.rng.clone());
                *fit = F::zero();
            }
        }

        if self.rng.gen::<f32>() < self.parameters.mutation_rate {
            if let Some(&mut (_, ref mut ind)) = self.rng.choose_mut(self.population.as_mut_slice()) {
                ind.mutate(&mut self.rng);
            }
        }

        self.population.first().cloned()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
