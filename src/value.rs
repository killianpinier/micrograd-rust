use std::{
    cell::RefCell,
    iter::Sum,
    ops::{Add, Deref, Mul, Sub},
    rc::Rc,
};

#[derive(Debug)]
pub enum Operation {
    Add(ValueWrapper, ValueWrapper),
    Mul(ValueWrapper, ValueWrapper),
    Pow(ValueWrapper, f32),
    Tanh(ValueWrapper),
}

#[derive(Debug, Clone)]
pub struct ValueWrapper(Rc<RefCell<Value>>);

#[derive(Debug)]
pub struct Value {
    data: f32,
    grad: f32,
    operation: Option<Operation>,
}

impl Value {
    fn new(data: f32, operation: Option<Operation>) -> Self {
        Self {
            data,
            grad: 0.0,
            operation,
        }
    }
}

impl ValueWrapper {
    pub fn new(data: f32, operation: Option<Operation>) -> Self {
        Self(Rc::new(RefCell::new(Value::new(data, operation))))
    }

    pub fn data(&self) -> f32 {
        self.borrow().data
    }

    pub fn update_data(&self, learning_step: f32) {
        let grad = self.borrow().grad;
        self.borrow_mut().data -= learning_step * grad;
    }

    pub fn reset_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn loss(self, pred: Self) -> Self {
        (pred - self).pow(2.0)
    }

    pub fn backpropagate(&mut self) {
        self.borrow_mut().grad = 1.0;
        self.backward();
    }

    fn backward(&mut self) {
        let grad = self.borrow().grad;
        if let Some(op) = &mut self.borrow_mut().operation {
            op.backward(grad);
        }
    }

    fn set_grad_and_backpropagate(&mut self, grad: f32) {
        self.borrow_mut().grad += grad;
        self.backward();
    }

    pub fn pow(self, power: f32) -> Self {
        Self::new(
            self.data().powf(power),
            Some(Operation::Pow(self.clone(), power)),
        )
    }

    pub fn tanh(self) -> Self {
        Self::new(self.data().tanh(), Some(Operation::Tanh(self.clone())))
    }
}

impl Add for ValueWrapper {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let inner = Rc::new(RefCell::new(Value::new(
            self.data() + rhs.data(),
            Some(Operation::Add(self.clone(), rhs.clone())),
        )));
        Self(inner)
    }
}

impl Sub for ValueWrapper {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let inner = Rc::new(RefCell::new(Value::new(
            self.data() - rhs.data(),
            Some(Operation::Add(
                self.clone(),
                ValueWrapper::new(-1.0, None) * rhs.clone(),
            )),
        )));
        Self(inner)
    }
}

impl Mul for ValueWrapper {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let inner = Rc::new(RefCell::new(Value::new(
            self.data() * rhs.data(),
            Some(Operation::Mul(self.clone(), rhs.clone())),
        )));
        Self(inner)
    }
}

impl Sum for ValueWrapper {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut result = iter.next().expect("Cannot sum items of an empty iterator");
        while let Some(val) = iter.next() {
            result = result + val;
        }
        result
    }
}

impl From<f32> for ValueWrapper {
    fn from(value: f32) -> Self {
        Self::new(value, None)
    }
}

impl Deref for ValueWrapper {
    type Target = Rc<RefCell<Value>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Operation {
    pub fn backward(&mut self, out_grad: f32) {
        match self {
            Self::Add(lhs, rhs) => {
                lhs.set_grad_and_backpropagate(out_grad);
                rhs.set_grad_and_backpropagate(out_grad);
            }
            Self::Mul(lhs, rhs) => {
                lhs.set_grad_and_backpropagate(rhs.borrow().data * out_grad);
                rhs.set_grad_and_backpropagate(lhs.borrow().data * out_grad);
            }
            Self::Pow(base, power) => {
                base.set_grad_and_backpropagate(*power * base.data().powf(*power - 1.0) * out_grad);
            }
            Self::Tanh(value) => {
                value.set_grad_and_backpropagate((1.0 - value.data().tanh().powi(2)) * out_grad)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::ValueWrapper;

    #[test]
    fn add_values() {
        let w1 = ValueWrapper::from(2.0);
        let x1 = ValueWrapper::from(5.0);
        let w2 = ValueWrapper::from(3.0);
        let x2 = ValueWrapper::from(6.0);
        let b = ValueWrapper::from(10.0);
        let w1x1 = w1 * x1;
        let w2x2 = w2 * x2;
        let w1x1w2x2 = w1x1 + w2x2;
        let n = w1x1w2x2 + b;

        assert_eq!(n.data(), 38.0)
    }

    #[test]
    fn add_two_same_values() {
        let a = ValueWrapper::from(1.0);
        let b = ValueWrapper::from(2.0);
        let c = ValueWrapper::from(3.0);

        let a_b = a.clone() + b;
        let a_c = a.clone() + c;

        let mut result = a_b + a_c;
        result.borrow_mut().grad = 1.0;
        result.backpropagate();
        println!("{:#?}", result);

        assert_eq!(a.clone().0.borrow().grad, 2.0);
    }

    #[test]
    fn test_backpropagation() {
        let w1 = ValueWrapper::from(-3.0);
        let x1 = ValueWrapper::from(2.0);
        let w2 = ValueWrapper::from(1.0);
        let x2 = ValueWrapper::from(0.0);
        let b = ValueWrapper::from(6.8813735870195432);
        let w1x1 = w1.clone() * x1.clone();
        let w2x2 = w2.clone() * x2.clone();
        let w1x1w2x2 = w1x1 + w2x2;
        let n = w1x1w2x2 + b;

        let mut o = n.tanh();
        o.backpropagate();

        assert_eq!(format!("{:.4}", o.data()), "0.7071");
        assert_eq!(format!("{:.5}", w1.borrow().grad), "1.00000");
        assert_eq!(format!("{:.5}", w2.borrow().grad), "0.00000");
        assert_eq!(format!("{:.5}", x1.borrow().grad), "-1.50000");
        assert_eq!(format!("{:.5}", x2.borrow().grad), "0.50000");
    }

    #[test]
    fn test_loss_backpropagation() {
        let w1 = ValueWrapper::from(-3.0);
        let x1 = ValueWrapper::from(2.0);
        let w2 = ValueWrapper::from(1.0);
        let x2 = ValueWrapper::from(0.0);
        let b = ValueWrapper::from(6.8813735870195432);
        let w1x1 = w1.clone() * x1.clone();
        let w2x2 = w2.clone() * x2.clone();
        let w1x1w2x2 = w1x1 + w2x2;
        let n = w1x1w2x2 + b;

        let o = n.clone().tanh();
        let mut loss = o.clone().loss(ValueWrapper::from(1.0));
        loss.backpropagate();

        let w1_grad = 2.0 * (o.data() - 1.0) * (1.0 - o.data().powi(2)) * x1.data();
        let w2_grad = 2.0 * (o.data() - 1.0) * (1.0 - o.data().powi(2)) * x2.data();
        let x1_grad = 2.0 * (o.data() - 1.0) * (1.0 - o.data().powi(2)) * w1.data();
        let x2_grad = 2.0 * (o.data() - 1.0) * (1.0 - o.data().powi(2)) * w2.data();

        assert_eq!(x1_grad, x1.borrow().grad);
        assert_eq!(x2_grad, x2.borrow().grad);

        assert_eq!(w1_grad, w1.borrow().grad);
        assert_eq!(w2_grad, w2.borrow().grad);
    }

    #[test]
    fn test_power_operation() {
        let x1 = ValueWrapper::from(2.0);
        let x2 = ValueWrapper::from(3.0);

        let mut result = (x1.clone() - x2.clone()).pow(2.0);
        result.backpropagate();

        assert_eq!(x1.clone().borrow().grad, 2.0 * (x1.data() - x2.data()))
    }
}
