extern crate rand;
extern crate calco;

use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{IndependentSample, Range};
use calco::Calco;
use calco::traits::{Evaluate, Mutate, Reproduce};

const SEED: [usize; 4] = [4, 2, 42, 4242];
const GLOBAL_MINIMUM: (f32, f32) = (15.0, -15.0);

#[derive(Debug, Copy, Clone)]
struct SimpleIndividual {
    x: f32,
    y: f32
}

impl Evaluate for SimpleIndividual {
    fn evaluate(&self) -> f32 {
        // note: inverse euclidean distance
        let (gx, gy) = GLOBAL_MINIMUM;
        let x = self.x - gx;
        let y = self.y - gy;
        let fit = 1.0 / (x * x + y * y).sqrt();
        if fit == 1.0 / 0.0 {
            println!("INFINITY");
        }
        fit
    }
}

impl Mutate for SimpleIndividual {
    fn mutate<R: Rng>(&mut self, rng: &mut R) {
        match rng.gen() { // ugly but... why not
            true => self.x = rng.gen_range(-10000.0, 10000.0),
            false => self.y = rng.gen_range(-10000.0, 10000.0),
        }
    }
}

impl Reproduce for SimpleIndividual {
    fn reproduce<'a, R: Rng>(&self, father: &Self, rng: &mut R) -> Vec<Self> {
        let mut children = Vec::with_capacity(2);
        match rng.gen_range(0, 2u8) {
            0 => {
                children.push(SimpleIndividual { x: self.x, y: father.y });
                children.push(SimpleIndividual { x: father.x, y: self.y });
            },
            1 => {
                children.push(SimpleIndividual {
                    x: (self.x + father.x) / 2.0,
                    y: (self.y + father.y) / 2.0
                });
                children.push(SimpleIndividual {
                    x: (self.x - father.x) / 2.0,
                    y: (self.y - father.y) / 2.0
                });
            },
            _ => {
                children.push(SimpleIndividual {
                    x: self.x + father.x,
                    y: self.y + father.y
                });
                children.push(SimpleIndividual {
                    x: self.x - father.x,
                    y: self.y - father.y
                });
            }
        }
        children
    }
}

fn main() {
    let mut rng = StdRng::from_seed(&SEED);
    let dist = Range::new(-1000.0, 1000.0);

    let calco: Calco<_, _> = (0..100).into_iter().map(|_| SimpleIndividual {
                                    x: dist.ind_sample(&mut rng),
                                    y: dist.ind_sample(&mut rng)
                                }).collect();

    for (gen, best) in calco.enumerate().take(100000).filter(|&(gen, _)| gen % 100 == 0) {
        println!("gen: {:?}, best: {:?}", gen, best);
    }
}
