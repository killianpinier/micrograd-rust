use core::panic;
use std::ops::Deref;

use crate::value::ValueWrapper;
use rand::{self, Rng};

#[derive(Debug)]
struct Neuron {
    weights: Vec<ValueWrapper>,
    b: ValueWrapper,
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct Entry(Vec<ValueWrapper>);
#[derive(Debug)]
pub struct TrainingSet(Vec<(Entry, ValueWrapper)>);

impl Neuron {
    fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: (0..nin)
                .map(|_| ValueWrapper::from(rng.gen_range(-1.0..1.0)))
                .collect(),
            b: ValueWrapper::from(rng.gen_range(-1.0..1.0)),
        }
    }

    fn calculate(&self, entries: &Entry) -> ValueWrapper {
        if self.weights.len() != entries.len() {
            panic!("Entries must have the same length as the number of nodes");
        }

        let result: ValueWrapper = self
            .weights
            .iter()
            .zip(entries.iter())
            .map(|(w, x)| w.clone() * x.clone())
            .sum();

        (result + self.b.clone()).tanh()
    }

    fn parameters(&self) -> Vec<ValueWrapper> {
        let mut parameters = self.weights.clone(); // Cloning pointers, not the Values themselves
        parameters.push(self.b.clone());
        parameters
    }
}

impl Layer {
    fn new(nin: usize, nout: usize) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    fn run(&self, entries: &Entry) -> Entry {
        Entry(self.neurons.iter().map(|n| n.calculate(entries)).collect())
    }

    fn parameters(&self) -> Vec<ValueWrapper> {
        let mut parameters = Vec::new();
        self.neurons.iter().for_each(|n| {
            parameters.append(&mut n.parameters());
        });
        parameters
    }
}

impl MLP {
    pub fn new(nin: usize, mut nouts: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let nouts_len = nouts.len();
        nouts.insert(0, nin);

        for i in 0..nouts_len {
            layers.push(Layer::new(nouts[i], nouts[i + 1]))
        }

        Self { layers }
    }

    pub fn train(
        &mut self,
        training_set: &TrainingSet,
        gradient_descent_step: f32,
        gradient_descent_count: usize,
    ) {
        for i in 0..gradient_descent_count {
            let mut parameters = self.get_parameters();
            let loss = self.loss(training_set);
            if i < 15 {
                println!("{}", loss.data());
            }
            self.backpropagate(loss, &mut parameters);
            self.update_network(parameters, gradient_descent_step);
        }
    }

    pub fn run(&mut self, entries: &Entry) -> ValueWrapper {
        let mut outputs: Entry = Entry::new();
        let mut layer_iter = self.layers.iter();

        // Instead of cloning entries and store it into outputs, we use it for the first layer, then we iterate over the
        // remaning ones.
        if let Some(first_layer) = layer_iter.next() {
            outputs = first_layer.run(entries);
        }
        while let Some(layer) = layer_iter.next() {
            outputs = layer.run(&outputs);
        }

        outputs
            .first()
            .expect("This program does not handle more than one neuron for the last layer.")
            .clone()
    }

    fn backpropagate(&mut self, mut loss: ValueWrapper, parameters: &mut Vec<ValueWrapper>) {
        self.reset_grads(parameters);
        loss.backpropagate();
    }

    fn update_network(&mut self, parameters: Vec<ValueWrapper>, learning_step: f32) {
        parameters.into_iter().for_each(|p| {
            //let mut value = p.borrow_mut();
            p.update_data(learning_step);
            //value.data += -1.0 * gradient_descent_step * value.grad;
        })
    }

    fn loss(&mut self, training_set: &TrainingSet) -> ValueWrapper {
        training_set
            .iter()
            .map(|(entry, pred)| {
                let result = self.run(entry);
                let loss = result.loss(pred.clone());
                loss
            })
            .sum()
    }

    fn get_parameters(&self) -> Vec<ValueWrapper> {
        let mut parameters = Vec::new();
        self.layers.iter().for_each(|l| {
            parameters.append(&mut l.parameters());
        });
        parameters
    }

    fn reset_grads(&self, parameters: &mut Vec<ValueWrapper>) {
        parameters.iter_mut().for_each(|p| {
            p.reset_grad();
            // p.borrow_mut().grad = 0.0;
        });
    }
}

impl Entry {
    fn new() -> Self {
        Self(Vec::new())
    }
}
impl From<Vec<f32>> for Entry {
    fn from(value: Vec<f32>) -> Self {
        Self(
            value
                .into_iter()
                .map(|data| ValueWrapper::from(data))
                .collect(),
        )
    }
}

impl Deref for Entry {
    type Target = Vec<ValueWrapper>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<(Vec<f32>, f32)>> for TrainingSet {
    fn from(value: Vec<(Vec<f32>, f32)>) -> Self {
        Self(
            value
                .into_iter()
                .map(|training_data| {
                    (
                        Entry::from(training_data.0),
                        ValueWrapper::from(training_data.1),
                    )
                })
                .collect(),
        )
    }
}

impl Deref for TrainingSet {
    type Target = Vec<(Entry, ValueWrapper)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write};

    use crate::value::Operation;

    use super::*;

    #[test]
    fn simple_mlp() {
        let mut mlp = MLP::new(3, vec![4, 4, 1]);

        let entries = Entry(vec![2.0.into(), 3.0.into(), (-1.0).into()]);
        let output = mlp.run(&entries);

        let mut current_entries: Vec<f32> = entries.iter().map(|entry| entry.data()).collect();

        for layer in mlp.layers {
            let mut entries: Vec<f32> = Vec::new();
            for neuron in layer.neurons {
                let mut temp = 0.0;
                for (x, w) in current_entries.iter().zip(neuron.weights.into_iter()) {
                    let mul = w.data() * x;
                    temp += mul;
                }
                temp += neuron.b.data();
                entries.push(temp.tanh());
            }
            current_entries = entries;
        }

        assert_eq!(current_entries.first().unwrap().clone(), output.data())
    }

    #[test]
    fn test_loss_function() {
        let mut mlp = MLP::new(3, vec![4, 4, 1]);

        let entries = Entry(vec![2.0.into(), 3.0.into(), (-1.0).into()]);
        let output = mlp.run(&entries);

        let mut current_entries: Vec<f32> = entries.iter().map(|entry| entry.data()).collect();

        for layer in mlp.layers {
            let mut entries: Vec<f32> = Vec::new();
            for neuron in layer.neurons {
                let mut temp = 0.0;
                for (x, w) in current_entries.iter().zip(neuron.weights.into_iter()) {
                    let mul = w.data() * x;
                    temp += mul;
                }
                temp += neuron.b.data();
                entries.push(temp.tanh());
            }
            current_entries = entries;
        }

        let calculated_loss = (current_entries.first().unwrap().clone() - 1.0).powi(2);
        assert_eq!(calculated_loss, output.loss(ValueWrapper::from(1.0)).data())
    }
}
