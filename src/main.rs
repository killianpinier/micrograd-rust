use std::time::SystemTime;

use micrograd_rs::nn::{TrainingSet, MLP};

fn main() {
    let mut mlp = MLP::new(3, vec![4, 4, 1]);
    let training_set = vec![
        (vec![2.0, 3.0, -1.0], 1.0),
        (vec![3.0, -1.0, 0.5], -1.0),
        (vec![0.5, 1.0, 1.0], -1.0),
        (vec![1.0, 1.0, -1.0], 1.0),
    ];
    let training_set = TrainingSet::from(training_set);

    let start = SystemTime::now();
    mlp.train(&training_set, 0.1, 100);

    println!("{:?}", start.elapsed().unwrap());

    for training_data in training_set.iter() {
        println!("{}", mlp.run(&training_data.0).get_data());
    }
}
