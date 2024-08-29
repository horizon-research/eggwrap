pub mod language;
use egg::*;
use crate::language::*;
use std::time::{Duration};
use pyo3::prelude::*;

type MyRunner = Runner<TnsrLang, TnsrAnalysis, ()>;

#[pyfunction]
fn greedy_optimize(program: String, n_sec: i32) -> String {
    let program_expr: RecExpr<TnsrLang> = program.parse().unwrap();
    let runner = MyRunner::new(Default::default())
                    .with_node_limit(10000000)
                    .with_time_limit(Duration::from_secs(n_sec as u64))
                    .with_iter_limit(100)
                    .with_expr(&program_expr);
    let runner = runner.run(&rules::<TnsrAnalysis>());
    let (egraph, root) = (runner.egraph, runner.roots[0]);
    let tnsr_cost = TnsrCost {
        egraph: &egraph,
    };
    let extractor = Extractor::new(&egraph, tnsr_cost);
    let (_, best) = extractor.find_best(root);
    return best.pretty(40 as usize);
}

#[pyfunction]
fn lp_optimize(program: String, n_sec: i32) -> String {
    let program_expr: RecExpr<TnsrLang> = program.parse().unwrap();
    let runner = MyRunner::new(Default::default())
                    .with_node_limit(10000000)
                    .with_time_limit(Duration::from_secs(n_sec as u64))
                    .with_iter_limit(100)
                    .with_expr(&program_expr);
    let runner = runner.run(&rules::<TnsrAnalysis>());

    let num_iter_sat = runner.iterations.len() - 1;

    // Print equality saturation stats
    // COMMENT THIS OUT LATER
    // println!("Runner complete!");
    // println!("  Nodes: {}", runner.egraph.total_size());
    // println!("  Classes: {}", runner.egraph.number_of_classes());
    // println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    // println!("  Number of iterations: {:?}", num_iter_sat);
    //
    // let (_, _, avg_nodes_per_class, num_edges, num_programs) =
    // get_stats(&runner.egraph);
    // println!("  Average nodes per class: {}", avg_nodes_per_class);
    // println!("  Number of edges: {}", num_edges);
    // println!("  Number of programs: {}", num_programs);


    let (egraph, root) = (runner.egraph, runner.roots[0]);
    let lp_tnsr_cost = TnsrCost {
        egraph: &egraph,
    };
    let mut lp_extractor = LpExtractor::new(&egraph, lp_tnsr_cost);
    let lp_best = lp_extractor.solve(root);
    lp_best.pretty(40 as usize)
}

#[pymodule]
fn eggwrap(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greedy_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(lp_optimize, m)?)?;
    Ok(())
}

fn get_stats(egraph: &EGraph<TnsrLang, TnsrAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}