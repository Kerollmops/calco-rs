extern crate rand;
extern crate calco;

use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{IndependentSample, Range};
use calco::{Calco, Parameters};
use calco::traits::{Evaluate, Mutate, Reproduce};

const SEED: [usize; 4] = [4, 2, 42, 4242];
const GLOBAL_MINIMUM: (f32, f32) = (15.0, -15.0);

#[derive(Debug, Copy, Clone)]
struct SimpleIndividual {
    x: f32,
    y: f32
}

impl Evaluate<f32> for SimpleIndividual {
    fn evaluate(&self) -> f32 {
        // note: inverse euclidean distance
        let (gx, gy) = GLOBAL_MINIMUM;
        let x = self.x - gx;
        let y = self.y - gy;
        let fit = 1.0 / (x * x + y * y).sqrt();
        // if fit == 1.0 / 0.0 { panic!("inf reached!"); }
        fit
    }
}

impl Mutate for SimpleIndividual {
    fn mutate<R: Rng>(&mut self, mut rng: R) {
        if rng.gen() {
            self.x = rng.gen_range(-10000.0, 10000.0)
        }
        else {
            self.y = rng.gen_range(-10000.0, 10000.0)
        }
    }
}

impl Reproduce for SimpleIndividual {
    fn reproduce<'a, R: Rng>(&self, father: &Self, mut rng: R) -> Self {
        match rng.gen_range(0, 6u8) {
            0 => SimpleIndividual { x: self.x, y: father.y },
            1 => SimpleIndividual { x: father.x, y: self.y },
            2 => SimpleIndividual {
                x: (self.x + father.x) / 2.0,
                y: (self.y + father.y) / 2.0
            },
            3 => SimpleIndividual {
                x: (self.x - father.x) / 2.0,
                y: (self.y - father.y) / 2.0
            },
            4 => SimpleIndividual {
                x: self.x + father.x,
                y: self.y + father.y
            },
            5 => SimpleIndividual {
                x: self.x - father.x,
                y: self.y - father.y
            },
            _ => unreachable!()
        }
    }
}

fn main() {
    let mut rng = StdRng::from_seed(&SEED);
    let dist = Range::new(-1000.0, 1000.0);

    let parameters = Parameters {
        mutation_rate: 0.2,
        population_limit: Some(100),
        elite_threshold: 0.9,
        deletion_threshold: 0.5,
    };

    let population = (0..100).into_iter()
                        .map(move |_| SimpleIndividual {
                            x: dist.ind_sample(&mut rng),
                            y: dist.ind_sample(&mut rng)
                        });

    let calco = Calco::with_rng(rng, parameters, population);

    for (gen, best) in calco.enumerate().take(100000)
                            .filter(|&(gen, _)| gen % 100 == 0) {
        println!("gen: {:?}, best: {:?}", gen, best);
    }
}
