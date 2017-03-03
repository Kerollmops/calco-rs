use rand::Rng;

pub trait Mutate {
    fn mutate<R: Rng>(&mut self, rng: R);
}
