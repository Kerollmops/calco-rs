use num::Num;

mod mutate;
mod reproduce;
mod evaluate;

pub use self::mutate::Mutate;
pub use self::reproduce::Reproduce;
pub use self::evaluate::Evaluate;

pub trait Individual<F: Num>: Evaluate<F> + Reproduce + Mutate { }

impl<F: Num, T: Evaluate<F> + Reproduce + Mutate> Individual<F> for T { }
