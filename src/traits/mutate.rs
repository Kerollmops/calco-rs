use rand::Rng;

pub trait Mutate {
    fn mutate<R: Rng>(&mut self, rng: &mut R); // -> Result<()>
}
