pub mod nn;
pub mod value;

// Usage:
//
// fn main() {
//     let mut mlp = MLP::new(3, vec![4, 4, 1]);
//     let training_set = vec![
//         (vec![2.0, 3.0, -1.0], 1.0),
//         (vec![3.0, -1.0, 0.5], -1.0),
//         (vec![0.5, 1.0, 1.0], -1.0),
//         (vec![1.0, 1.0, -1.0], 1.0),
//     ];
//     let training_set = TrainingSet::from(training_set);
//
//     mlp.train(&training_set, 0.1, 100);
//
//     for training_data in training_set.iter() {
//         println!("{}", mlp.run(&training_data.0).data());
//     }
// }
