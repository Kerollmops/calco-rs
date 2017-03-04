use rand::Rng;

pub trait Reproduce: Sized {
    fn reproduce<R: Rng>(&self, father: &Self, rng: R) -> Self;
}
