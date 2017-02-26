use rand::Rng;

pub trait Reproduce: Sized {
    fn reproduce<'a, R: Rng>(&self, father: &Self, rng: &mut R) -> Vec<Self>;
}
