//! A Little implementation of the roulette-wheel principle, `RouletteWheel<T>`.
//! https://wikipedia.org/wiki/Fitness_proportionate_selection
//!
//! ![Fitness proportionate selection](https://upload.wikimedia.org/wikipedia/commons/2/2a/Fitness_proportionate_selection_example.png)
//!
//! # Examples usages
//!
//! ```
//! use roulette_wheel::RouletteWheel;
//!
//! fn evaluate(individual: &i32) -> f32 { *individual as f32 } // mmm...!
//!
//! let population: Vec<_> = (1..10).into_iter().collect();
//! let fitnesses: Vec<_> = population.iter().map(|ind| evaluate(ind)).collect();
//!
//! let rw: RouletteWheel<_> = fitnesses.into_iter().zip(population).collect();
//!
//! // let's collect the individuals in the order in which the roulette wheel gives them
//! let individuals: Vec<_> = rw.into_iter().map(|(_, ind)| ind).collect();
//! // rw.select_iter() will not consume the roulette wheel
//! // while rw.into_iter() will !
//!
//! fn crossover(mother: &i32, _father: &i32) -> i32 { mother.clone() } // unimplemented!()
//!
//! // now merge each individual by couples
//! let new_population: Vec<_> = individuals.chunks(2)
//!                                  .filter(|couple| couple.len() == 2)
//!                                  .map(|couple| {
//!                                       let (mother, father) = (couple[0], couple[1]);
//!                                       crossover(&mother, &father)
//!                                       // note: for this example we return only one individual,
//!                                       //       the population will shrink
//!                                       //       .flat_map() can resolve this issue
//!                                   }).collect();
//! ```

extern crate rand;
extern crate num;

use std::iter::{Iterator, FromIterator, IntoIterator, Sum};
use rand::{Rng, ThreadRng, thread_rng};
use std::cmp::PartialOrd;
use rand::distributions::{Range, IndependentSample};
use rand::distributions::range::SampleRange;
use num::{Num, ToPrimitive, FromPrimitive};

/// A roulette-wheel container
pub struct RouletteWheel<F: Num, T> {
    total_fitness: F,
    fitnesses: Vec<F>,
    population: Vec<T>
}

impl<F: Num + Copy, T: Clone> Clone for RouletteWheel<F, T> {
    fn clone(&self) -> RouletteWheel<F, T> {
        RouletteWheel {
            total_fitness: self.total_fitness,
            fitnesses: self.fitnesses.clone(),
            population: self.population.clone()
        }
    }
}

impl<F: Num + Copy + Sum<F>, T> FromIterator<(F, T)> for RouletteWheel<F, T> {
    /// Don't add zero or negative values !
    fn from_iter<A>(iter: A) -> Self where A: IntoIterator<Item=(F, T)> {
        let (fitnesses, population): (Vec<F>, _) = iter.into_iter().unzip();
        let total_fitness = fitnesses.iter().cloned().sum();
        RouletteWheel {
            total_fitness: total_fitness,
            fitnesses: fitnesses,
            population: population
        }
    }
}

impl<F: Num + Copy, T> RouletteWheel<F, T> {
    /// create a new empty random-wheel.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let rw = RouletteWheel::<u8>::new();
    /// ```
    pub fn new() -> RouletteWheel<F, T> {
        RouletteWheel {
            total_fitness: F::zero(),
            fitnesses: Vec::new(),
            population: Vec::new()
        }
    }

    /// Creates an empty RouletteWheel with space for at least n elements.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let rw = RouletteWheel::<u8>::with_capacity(15);
    ///
    /// assert_eq!(rw.len(), 0);
    /// ```
    pub fn with_capacity(cap: usize) -> RouletteWheel<F, T> {
        RouletteWheel {
            total_fitness: F::zero(),
            fitnesses: Vec::with_capacity(cap),
            population: Vec::with_capacity(cap)
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    /// The collection may reserve more space to avoid frequent reallocations.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let mut rw = RouletteWheel::<u8>::new();
    /// rw.reserve(20);
    ///
    /// assert_eq!(rw.len(), 0);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.fitnesses.reserve(additional);
        self.population.reserve(additional);
    }

    /// Returns the number of elements in the wheel.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let rw: RouletteWheel<_> = [(0.1, 10), (0.2, 15), (0.5, 20)].iter().cloned().collect();
    ///
    /// assert_eq!(rw.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.population.len()
    }

    /// Returns `true` if empty else return `false`.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let empty_rw = RouletteWheel::<u8>::new();
    ///
    /// assert_eq!(empty_rw.is_empty(), true);
    ///
    /// let non_empty_rw: RouletteWheel<_> = [(0.1, 10), (0.2, 15), (0.5, 20)].iter().cloned().collect();
    ///
    /// assert_eq!(non_empty_rw.is_empty(), false);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.population.is_empty()
    }

    /// Remove all elements in this wheel.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let mut rw: RouletteWheel<_> = [(0.1, 10), (0.2, 15), (0.5, 20)].iter().cloned().collect();
    ///
    /// assert_eq!(rw.len(), 3);
    ///
    /// rw.clear();
    ///
    /// assert_eq!(rw.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.fitnesses.clear();
        self.population.clear();
    }

    /// Add an element associated with a probability.
    ///
    /// # Panics
    ///
    /// This function might panic if the fitness is less than zero
    /// or if the total fitness gives a non-finite fitness (`Inf`).
    ///
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let mut rw = RouletteWheel::new();
    ///
    /// rw.push(1.0, 'r');
    /// rw.push(1.0, 'c');
    /// rw.push(1.0, 'a');
    ///
    /// assert_eq!(rw.len(), 3);
    /// ```
    pub fn push(&mut self, fitness: F, individual: T) where F: PartialOrd {
        assert!(fitness >= F::zero(), "Can't push less than or equal to zero fitness");
        unsafe { self.unchecked_push(fitness, individual) }
    }

    /// Add an element associated with a probability.
    /// This unsafe function doesn't check for total fitness overflow
    /// nether fitness positivity.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let mut rw = RouletteWheel::new();
    ///
    /// unsafe { rw.unchecked_push(1.0, 'r') };
    /// unsafe { rw.unchecked_push(1.0, 'c') };
    /// unsafe { rw.unchecked_push(1.0, 'a') };
    ///
    /// assert_eq!(rw.len(), 3);
    /// ```
    pub unsafe fn unchecked_push(&mut self, fitness: F, individual: T) {
        self.total_fitness = self.total_fitness + fitness;
        self.fitnesses.push(fitness);
        self.population.push(individual);
    }

    /// Returns the sum of all individual fitnesses.
    /// # Example
    ///
    /// ```
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let mut rw = RouletteWheel::new();
    ///
    /// rw.push(3.0, 'r');
    /// rw.push(2.0, 'c');
    /// rw.push(1.5, 'a');
    ///
    /// assert_eq!(rw.total_fitness(), 6.5);
    /// ```
    pub fn total_fitness(&self) -> F {
        self.total_fitness
    }

    /// Returns an iterator over the RouletteWheel.
    ///
    /// # Examples
    ///
    /// ``` ignore
    /// use roulette_wheel::RouletteWheel;
    ///
    /// let rw: RouletteWheel<_> = [(0.1, 10), (0.2, 15), (0.5, 20)].iter().cloned().collect();
    /// let mut iterator = rw.select_iter();
    ///
    /// assert_eq!(iterator.next(), Some((0.5, &20)));
    /// assert_eq!(iterator.next(), Some((0.1, &10)));
    /// assert_eq!(iterator.next(), Some((0.2, &15)));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn select_iter(&self) -> SelectIter<ThreadRng, F, T>
        where F: SampleRange + ToPrimitive + FromPrimitive {
        SelectIter::<ThreadRng, _, _>::new(&self)
    }
}

/// Immutable RouletteWheel iterator
///
/// This struct is created by the [`select_iter`].
///
/// [`iter`]: struct.RouletteWheel.html#method.select_iter
pub struct SelectIter<'a, R, F, T>
    where R: Rng, F: 'a + Num + Copy + SampleRange + ToPrimitive + FromPrimitive, T: 'a {

    distribution_range: Range<f64>,
    rng: R,
    total_fitness: F,
    fitnesses_ids: Vec<(usize, F)>,
    roulette_wheel: &'a RouletteWheel<F, T>
}

impl<'a, R, F, T> SelectIter<'a, R, F, T>
    where R: Rng, F: Num + Copy + SampleRange + ToPrimitive + FromPrimitive {

    pub fn new(roulette_wheel: &'a RouletteWheel<F, T>) -> SelectIter<'a, ThreadRng, F, T> {
        SelectIter::with_rng(thread_rng(), roulette_wheel)
    }

    pub fn with_rng(rng: R, roulette_wheel: &'a RouletteWheel<F, T>) -> SelectIter<'a, R, F, T> {
        SelectIter {
            distribution_range: Range::new(0.0, 1.0),
            rng: rng,
            total_fitness: roulette_wheel.total_fitness,
            fitnesses_ids: roulette_wheel.fitnesses.iter().cloned().enumerate().collect(),
            roulette_wheel: roulette_wheel
        }
    }
}

impl<'a, R, F, T: 'a> Iterator for SelectIter<'a, R, F, T>
    where R: Rng, F: Num + Copy + SampleRange + PartialOrd + ToPrimitive + FromPrimitive {

    type Item = (F, &'a T);

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.fitnesses_ids.len(), Some(self.fitnesses_ids.len()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if !self.fitnesses_ids.is_empty() {
            let sample = self.distribution_range.ind_sample(&mut self.rng);
            let total_fitness = self.total_fitness.to_f64().expect("Can't create f64 from F value!");
            let mut selection = F::from_f64(sample * total_fitness).expect("Can't create F from f64 value!");

            let index = self.fitnesses_ids.iter().position(|&(_, fit)| {
                            selection = selection - fit;
                            selection <= F::zero()
                        }).expect("Can't select next index! (float precision?)");

            let (index, fitness) = self.fitnesses_ids.swap_remove(index);
            self.total_fitness = self.total_fitness - fitness;
            Some((fitness, &self.roulette_wheel.population[index]))
        }
        else { None }
    }
}

/// An iterator that moves out of a RouletteWheel.
///
/// This `struct` is created by the `into_iter` method on [`RouletteWheel`][`RouletteWheel`] (provided
/// by the [`IntoIterator`] trait).
///
/// [`RouletteWheel`]: struct.RouletteWheel.html
/// [`IntoIterator`]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
pub struct IntoSelectIter<R: Rng, F: Num + Copy + SampleRange, T> {
    distribution_range: Range<f64>,
    rng: R,
    total_fitness: F,
    fitnesses: Vec<F>,
    population: Vec<T>
}

impl<F, T> IntoIterator for RouletteWheel<F, T>
    where F: Num + Copy + SampleRange + PartialOrd + ToPrimitive + FromPrimitive {

    type Item = (F, T);
    type IntoIter = IntoSelectIter<ThreadRng, F, T>;

    fn into_iter(self) -> IntoSelectIter<ThreadRng, F, T> {
        IntoSelectIter::<ThreadRng, _, _>::new(self)
    }
}

impl<R: Rng, F: Num + Copy + SampleRange + PartialOrd, T> IntoSelectIter<R, F, T> {
    pub fn new(roulette_wheel: RouletteWheel<F, T>) -> IntoSelectIter<ThreadRng, F, T> {
        IntoSelectIter::with_rng(thread_rng(), roulette_wheel)
    }

    pub fn with_rng(rng: R, roulette_wheel: RouletteWheel<F, T>) -> IntoSelectIter<R, F, T> {
        IntoSelectIter {
            distribution_range: Range::new(0.0, 1.0),
            rng: rng,
            total_fitness: roulette_wheel.total_fitness,
            fitnesses: roulette_wheel.fitnesses,
            population: roulette_wheel.population
        }
    }
}

impl<R, F, T> Iterator for IntoSelectIter<R, F, T>
    where R: Rng, F: Num + Copy + SampleRange + PartialOrd + ToPrimitive + FromPrimitive {

    type Item = (F, T);

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.fitnesses.len(), Some(self.fitnesses.len()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if !self.fitnesses.is_empty() {
            let sample = self.distribution_range.ind_sample(&mut self.rng);
            let total_fitness = self.total_fitness.to_f64().expect("Can't create f64 from F value!");
            let mut selection = F::from_f64(sample * total_fitness).expect("Can't create F from f64 value!");

            let index = self.fitnesses.iter().position(|fit| {
                            selection = selection - *fit;
                            selection <= F::zero()
                        }).expect("Can't select next index! (float precision?)");

            let fitness = self.fitnesses.swap_remove(index);
            let individual = self.population.swap_remove(index);
            self.total_fitness = self.total_fitness - fitness;
            Some((fitness, individual))
        }
        else { None }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::StdRng;
    use {RouletteWheel, SelectIter, IntoSelectIter};

    const SEED: [usize; 4] = [4, 2, 42, 4242];

    #[test]
    fn test_select_iter_seeded() {
        let rng = StdRng::from_seed(&SEED);

        let fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5];
        let fitnesses = fitnesses.iter().cloned();
        let population = 15..20;
        let rw: RouletteWheel<_> = fitnesses.zip(population).collect();

        let mut iter = SelectIter::with_rng(&rw, rng);

        assert_eq!(iter.next(), Some((0.5, &19)));
        assert_eq!(iter.next(), Some((0.3, &17)));
        assert_eq!(iter.next(), Some((0.4, &18)));
        assert_eq!(iter.next(), Some((0.2, &16)));
        assert_eq!(iter.next(), Some((0.1, &15)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_select_iter_seeded() {
        let rng = StdRng::from_seed(&SEED);

        let fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5];
        let fitnesses = fitnesses.iter().cloned();
        let population = 15..20;
        let rw: RouletteWheel<_> = fitnesses.zip(population).collect();

        let mut iter = IntoSelectIter::with_rng(rw, rng);

        assert_eq!(iter.next(), Some((0.5, 19)));
        assert_eq!(iter.next(), Some((0.3, 17)));
        assert_eq!(iter.next(), Some((0.4, 18)));
        assert_eq!(iter.next(), Some((0.2, 16)));
        assert_eq!(iter.next(), Some((0.1, 15)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_len() {
        let mut rw = RouletteWheel::<u8>::new();

        assert_eq!(rw.len(), 0);

        rw.push(0.1, 1);
        rw.push(0.1, 1);
        rw.push(0.1, 1);
        rw.push(0.1, 1);

        assert_eq!(rw.len(), 4);
    }
}
