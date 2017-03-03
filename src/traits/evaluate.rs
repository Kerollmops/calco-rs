use num::Num;

pub trait Evaluate<F: Num> {
    fn evaluate(&self) -> F;
}
