pub trait Reproduce: Sized {
    fn reproduce<'a, I>(&self, &I) -> Vec<Self> where I: Iterator<Item=&'a Self>, Self: 'a;
}
