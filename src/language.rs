use egg::*;
use std::cmp::max;

define_language! {
    pub enum TnsrLang {
        "input"  = Input([Id; 1]), // Input T name@T/F@dim1_dim2_...
        "Add"    = Add([Id; 2]), // T x T --> T
        "Sub"    = Sub([Id; 2]), // T x T --> T
        "Neg"    = Neg([Id; 1]), // T --> T
        "Mul"    = Mul([Id; 2]), // T x T --> T
        "Div"    = Div([Id; 2]), // T x T --> T
        "smul"   = Smul([Id; 2]), // S x T --> T
        "MatMul" = MatMul([Id; 2]), // T x T --> T
        "Relu"   = Relu([Id; 1]), // Relu activation T --> T
        "Pow"    = Pow([Id; 2]), // Elementwise power T x S --> T
        "Transpose"  = Transpose([Id;1]), // T --> T
        "noop"  = Noop(Box<[Id]>), // Combines multiple outputs
        "fmul" = FMul([Id; 2]),
        "fadd" = FAdd([Id; 2]),
        "Floor" = Floor([Id; 1]),
        "Mod" = Mod([Id; 2]),
        "Max" = Max([Id; 3]),
        "Min" = Min([Id; 3]),
        Var(Symbol),
    }
}

// Enum defines the different types allowed in TnsLang
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name, // Input tensors
    Scalar, // Input scalars
    Tnsr, // Tensor
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

// Metadata struct for TensorAnalysis (additional info for an e-class)
#[derive(Debug, Clone)]
pub struct Metadata {
    pub dtype: DataKind, // The type of the e-class
    pub name: String, // Name of input if it is an input type
    pub dims: Vec<i32>,
    pub constant_foldable: bool,
    pub scalar: f64,
}

pub struct TnsrAnalysis {

}

impl Default for TnsrAnalysis {
    fn default() -> Self { TnsrAnalysis {

    } }
}

fn extract_dim(foo: &Metadata) -> Vec<i32> {
    if foo.dtype == DataKind::Tnsr {
        foo.dims.clone()
    } else if foo.dtype == DataKind::Scalar {
        vec![1]
    } else {
        vec![]
    }
}

// Metadata analysis
impl Analysis<TnsrLang> for TnsrAnalysis {
    type Data = Metadata;

    // Merges two metadata when two eclasses are merged
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        if from.constant_foldable && (!to.constant_foldable) {
            to.constant_foldable = false;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }

    fn make(egraph: &EGraph<TnsrLang, Self>, enode: &TnsrLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        // Parse name@T/F@dim1_dim2_... and return a tuple
        // (String, bool, Vec<i32>)
        let parse_input = |name: &Id| {
            let name_vec: Vec<&str> = x(name).name.split("@").collect();
            assert!(name_vec.len() == 3);
            let name: String = String::from(name_vec[0]);
            let is_constant: bool = name_vec[1] == "T";
            let dims: Vec<i32> = name_vec[2]
                .split("_")
                .map(|x| x.parse::<i32>().unwrap())
                .collect();
            (name, is_constant, dims)
        };

        match enode {
            TnsrLang::Input([name]) => {
                assert!(x(name).dtype == DataKind::Name || x(name).dtype == DataKind::Scalar);
                if x(name).dtype == DataKind::Name {
                    let parsed_input = parse_input(name);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        name: String::new(),
                        dims: parsed_input.2,
                        constant_foldable: parsed_input.1,
                        scalar: 0.0,
                    }
                } else {
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        name: String::new(),
                        dims: vec![1],
                        constant_foldable: true,
                        scalar: x(name).scalar,
                    }
                }
            },

            TnsrLang::Add([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr || x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                let (a_dim, b_dim) = (extract_dim(x(a)), extract_dim(x(b)));
                assert!(is_broadcastable(&a_dim, &b_dim));
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&a_dim, &b_dim),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Sub([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr || x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                let (a_dim, b_dim) = (extract_dim(x(a)), extract_dim(x(b)));
                assert!(is_broadcastable(&a_dim, &b_dim));
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&a_dim, &b_dim),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Neg([t]) => {
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(t).dims.clone(),
                    constant_foldable: x(t).constant_foldable,
                    scalar: 0.0,
                }
            }

            TnsrLang::Mul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr || x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                let (a_dim, b_dim) = (extract_dim(x(a)), extract_dim(x(b)));
                assert!(is_broadcastable(&a_dim, &b_dim));
                let foldable: bool = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&a_dim, &b_dim),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Div([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr || x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                let (a_dim, b_dim) = (extract_dim(x(a)), extract_dim(x(b)));
                assert!(is_broadcastable(&a_dim, &b_dim));
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&a_dim, &b_dim),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Smul([s, t]) => {
                assert!(x(s).dtype == DataKind::Scalar);
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(s).constant_foldable && x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(t).dims.clone(),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::MatMul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert_eq!(x(a).dims[x(a).dims.len()-1], x(b).dims[0]);
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: matmul_dimension(&x(a).dims, &x(b).dims),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Noop(children) => {
                let foldable = children.iter().all(|child| x(child).constant_foldable);
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: vec![],
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Relu([t]) => {
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(t).dims.clone(),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Pow([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                assert!(is_broadcastable(&x(a).dims, &x(b).dims));
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&x(a).dims, &x(b).dims),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Floor([a]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let foldable = x(a).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(a).dims.clone(),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Mod([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert!(is_broadcastable(&x(a).dims, &x(b).dims));
                let foldable = x(a).constant_foldable && x(b).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: broadcast_vectors(&x(a).dims, &x(b).dims),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Max([a, b, c]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert!(x(c).dtype == DataKind::Tnsr);
                let foldable = x(a).constant_foldable && x(b).constant_foldable && x(c).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(a).dims.clone(),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Min([a, b, c]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                assert!(x(c).dtype == DataKind::Tnsr);
                let foldable = x(a).constant_foldable && x(b).constant_foldable && x(c).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: x(a).dims.clone(),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::Transpose([t]) => {
                assert!(x(t).dtype == DataKind::Tnsr);
                let foldable = x(t).constant_foldable;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    name: String::new(),
                    dims: transpose_shape(&x(t).dims),
                    constant_foldable: foldable,
                    scalar: 0.0,
                }
            },

            TnsrLang::FMul([a, b]) => {
                assert!(x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Scalar);
                let val = x(a).scalar * x(b).scalar;
                Self::Data {
                    dtype: DataKind::Scalar,
                    name: val.to_string(),
                    dims: vec![],
                    constant_foldable: true,
                    scalar: val,
                }
            },

            TnsrLang::FAdd([a, b]) => {
                assert!(x(a).dtype == DataKind::Scalar);
                assert!(x(b).dtype == DataKind::Scalar);
                let val = x(a).scalar + x(b).scalar;
                Self::Data {
                    dtype: DataKind::Scalar,
                    name: val.to_string(),
                    dims: vec![],
                    constant_foldable: true,
                    scalar: val,
                }
            },

            TnsrLang::Var(_s) => {
                let num = _s.as_str().to_string().parse::<f64>();
                match num {
                    Ok(value) => {
                        Self::Data {
                            dtype: DataKind::Scalar,
                            name: _s.as_str().to_string(),
                            dims: vec![],
                            constant_foldable: true,
                            scalar: value,
                        }
                    }
                    Err(_) => {
                        //println!("{}", _s.as_str().to_string());
                        Self::Data {
                            dtype: DataKind::Name,
                            name: _s.as_str().to_string(),
                            dims: vec![],
                            constant_foldable: true,
                            scalar: 0.0,
                        }
                    }
                }
            },
        }
    }
}

fn transpose_shape(shape: &Vec<i32>) -> Vec<i32> {
    let mut transpose_shape = shape.clone();
    transpose_shape.reverse();
    transpose_shape
}

fn is_broadcastable(foo: &Vec<i32>, bar: &Vec<i32>) -> bool {
    if foo == bar {
        return true
    }
    let (shorter, longer) = if foo.len() <= bar.len() {
        (foo, bar)
    } else {
        (bar, foo)
    };

    let mut lengthened: Vec<i32> = shorter.clone();
    lengthened.reverse();
    lengthened.resize(longer.len(), 1);
    lengthened.reverse();
    
    for i in 0..longer.len() {
        if lengthened[i] != longer[i] && lengthened[i] != 1 && longer[i] != 1 {
            return false
        }
    }
    return true
}

fn broadcast_vectors(foo: &Vec<i32>, bar: &Vec<i32>) -> Vec<i32> {
    if !is_broadcastable(foo, bar) {
        return vec![];
    } else {
        let (shorter, longer) = if foo.len() <= bar.len() {
            (foo, bar)
        } else {
            (bar, foo)
        };

        let mut lengthened: Vec<i32> = shorter.clone();
        lengthened.reverse();
        lengthened.resize(longer.len(), 1);
        lengthened.reverse();

        let mut result: Vec<i32> = vec![0; longer.len()];
        for i in 0..longer.len() {
            result[i] = max(lengthened[i], longer[i]);
        }

        return result
    }
}

fn matmul_dimension(a: &Vec<i32>, b: &Vec<i32>) -> Vec<i32> {
    let mut a_copy = a.clone();
    a_copy.pop();
    if a_copy.is_empty() {
        a_copy = vec![1];
    }
    let mut b_copy = b.clone();
    b_copy.remove(0);
    if b_copy.is_empty() {
        b_copy = vec![1];
    }

    return [&a_copy[..], &b_copy[..]].concat();
}

pub fn rules<A: Analysis<TnsrLang>>() -> Vec<Rewrite<TnsrLang, TnsrAnalysis>> { vec![
    ////// Single-Operator Rules
    // Add
    rewrite!("add-associative"; "(Add ?x (Add ?y ?z))" => "(Add (Add ?x ?y) ?z)"),
    rewrite!("-add-associative"; "(Add (Add ?x ?y) ?z)" =>  "(Add ?x (Add ?y ?z))"),
    rewrite!("add-commutative"; "(Add ?x ?y)" => "(Add ?y ?x)"),
    // Neg
    // rewrite!("neg-is-its-inverse"; "(Neg (Neg ?x))" => "?x"),
    // rewrite!("-neg-is-its-inverse"; "?x" => "(Neg (Neg ?x))"),
    // Mul
    rewrite!("mul-associative"; "(Mul ?x (Mul ?y ?z))" => "(Mul (Mul ?x ?y) ?z)"),
    rewrite!("-mul-associative"; "(Mul (Mul ?x ?y) ?z)" => "(Mul ?x (Mul ?y ?z))"),
    rewrite!("mul-commutative"; "(Mul ?x ?y)" => "(Mul ?y ?x)"),
    // Transpose
    // rewrite!("transpose-is-its-own-inverse"; "(Transpose (Transpose ?x))" => "?x"),
    // rewrite!("-transpose-is-its-own-inverse"; "?x" => "(Transpose (Transpose ?x))"),
    // MatMul
    rewrite!("matmul-is-associative"; "(MatMul ?x (MatMul ?y ?z))" => "(MatMul (MatMul ?x ?y) ?z)"),
    rewrite!("-matmul-is-associative"; "(MatMul (MatMul ?x ?y) ?z)" => "(MatMul ?x (MatMul ?y ?z))"),

    // Neg with other operators
//     rewrite!("sub-is-adding-negative"; "(Sub ?x ?y)" => "(Add ?x (Neg ?y))"),
//     rewrite!("-sub-is-adding-negative"; "(Add ?x (Neg ?y))" => "(Sub ?x ?y)"),
//     rewrite!("sub-is-reversed-with-neg"; "(Sub ?x ?y)" => "(Neg (Sub ?y ?x))"),
//     rewrite!("-sub-is-reversed-with-neg"; "(Neg (Sub ?y ?x))" => "(Sub ?x ?y)"),
//     rewrite!("transpose-commutative-with-neg"; "(Transpose (Neg ?x))" => "(Neg (Transpose ?x))"),
//     rewrite!("-transpose-commutative-with-neg"; "(Neg (Transpose ?x))" => "(Transpose (Neg ?x))"),
//     rewrite!("mul-of-negs-cancel"; "(Mul (Neg ?x) (Neg ?y))" => "(Mul ?x ?y)"),
//     rewrite!("-mul-of-negs-cancel"; "(Mul ?x ?y)" => "(Mul (Neg ?x) (Neg ?y))"),
//     rewrite!("div-of-negs-cancel"; "(Div (Neg ?x) (Neg ?y))" => "(Div ?x ?y)"),
//     rewrite!("-div-of-negs-cancel"; "(Div ?x ?y)" => "(Div (Neg ?x) (Neg ?y))"),
//     rewrite!("neg-distributes-over-add"; "(Neg (Add ?x ?y))" => "(Sub (Neg ?x) ?y)"),
//     rewrite!("-neg-distributes-over-add"; "(Sub (Neg ?x) ?y)" => "(Neg (Add ?x ?y))"),
//     rewrite!("neg-associates-over-mul"; "(Mul (Neg ?x) ?y)" => "(Mul ?x (Neg ?y))"),
//     rewrite!("-neg-associates-over-mul"; "(Mul ?x (Neg ?y))" => "(Mul (Neg ?x) ?y)"),
//     rewrite!("neg-associates-over-div"; "(Div (Neg ?x) ?y)" => "(Div ?x (Neg ?y))"),
//     rewrite!("-neg-associates-over-div"; "(Div ?x (Neg ?y))" => "(Div (Neg ?x) ?y)"),

    // Mul && Smul
    rewrite!("mul-distributes-over-add"; "(Mul (Add ?x ?y) ?z)" => "(Add (Mul ?x ?z) (Mul ?y ?z))"),
    rewrite!("-mul-distributes-over-add"; "(Add (Mul ?x ?z) (Mul ?y ?z))" => "(Mul (Add ?x ?y) ?z)"),

    // Linearity of MatMul
    rewrite!("MatMul-is-linear-over-add"; "(MatMul ?x (Add ?y ?z))" => "(Add (MatMul ?x ?y) (MatMul ?x ?z))"),
    rewrite!("-MatMul-is-linear-over-add"; "(Add (MatMul ?x ?y) (MatMul ?x ?z))" => "(MatMul ?x (Add ?y ?z))"),

    // Relu & Transpose
    // rewrite!("relu-transpose"; "(Relu (Transpose ?x))" => "(Transpose (Relu ?x))"),
    // rewrite!("-relu-transpose"; "(Transpose (Relu ?x))" => "(Relu (Transpose ?x))"),

    // Transpose 
    // rewrite!("transpose-commutativity-ewadd"; "(Transpose (Add ?x ?y))" => "(Add (Transpose ?x)  (Transpose ?y))"),
    // rewrite!("-transpose-commutativity-ewadd"; "(Add (Transpose ?x)  (Transpose ?y))" => "(Transpose (Add ?x ?y))"),
    // rewrite!("transpose-commutativity-mul"; "(Transpose (Mul ?x ?y))" => "(Mul (Transpose ?x)  (Transpose ?y))"),
    // rewrite!("-transpose-commutativity-mul"; "(Mul (Transpose ?x)  (Transpose ?y))" => "(Transpose (Mul ?x ?y))"),
    // rewrite!("matmul-transpose"; "(Transpose (MatMul ?x ?y))" => "(MatMul (Transpose ?y)  (Transpose ?x))"),
    // rewrite!("-matmul-transpose"; "(MatMul (Transpose ?y)  (Transpose ?x))" => "(Transpose (MatMul ?x ?y))"),

    // Pow
    rewrite!("pow-mul"; "(Pow (Mul ?x ?y) ?z)" => "(Mul (Pow ?x ?z) (Pow ?y ?z))"),
    rewrite!("-pow-mul"; "(Mul (Pow ?x ?z) (Pow ?y ?z))" => "(Pow (Mul ?x ?y) ?z)"),
    rewrite!("pow-div"; "(Pow (Div ?x ?y) ?z)" => "(Div (Pow ?x ?z) (Pow ?y ?z))"),
    rewrite!("-pow-div"; "(Div (Pow ?x ?z) (Pow ?y ?z))" => "(Pow (Div ?x ?y) ?z)"),

    // Div
    rewrite!("div-distributes-over-add"; "(Div (Add ?x ?y) ?z)" => "(Add (Div ?x ?z) (Div ?y ?z))"),
    rewrite!("-div-distributes-over-add"; "(Add (Div ?x ?z) (Div ?y ?z))" => "(Div (Add ?x ?y) ?z)"),

    // Pow associative
    // rewrite!("pow-comm-l"; "(Pow ?x (Pow ?y ?z))" => "(Pow (Pow ?x ?y) ?z)"),
    // rewrite!("pow-comm-r"; "(Pow (Pow ?x ?y) ?z)" => "(Pow ?x (Pow ?y ?z))"),

    // Div-Mul associative
    // rewrite!("div-mul-comm-l"; "(Mul ?x (Div ?y ?z))" => "(Div (Mul ?x ?y) ?z)"),
    // rewrite!("div-mul-comm-r"; "(Div (Mul ?x ?y) ?z)" => "(Mul ?x (Div ?y ?z))"),
]}

// struct FMulApplier {
//     foo: Var,
//     bar: Var,
// }

impl Applier<TnsrLang, TnsrAnalysis> for FMulApplier {
    fn apply_one(&self, egraph: &mut EGraph<TnsrLang, TnsrAnalysis>, matched_id: Id, subst: &Subst, _: Option<&PatternAst<TnsrLang>>, _: Symbol) -> Vec<Id> {
        let foo_scalar = egraph[subst[self.foo]].data.scalar;
        let bar_scalar = egraph[subst[self.bar]].data.scalar;
        let new_scalar = foo_scalar * bar_scalar;
        let new_var = egraph.add(TnsrLang::Var(new_scalar.to_string().into()));
        let new_node = egraph.add(TnsrLang::Input([new_var]));
        if egraph.union(new_node, matched_id) {
            vec![new_node, matched_id]
        } else {
            vec![]
        }
    }
}

// struct FAddApplier {
//     foo: Var,
//     bar: Var,
// }

impl Applier<TnsrLang, TnsrAnalysis> for FAddApplier {
    fn apply_one(&self, egraph: &mut EGraph<TnsrLang, TnsrAnalysis>, matched_id: Id, subst: &Subst, _: Option<&PatternAst<TnsrLang>>, _: Symbol) -> Vec<Id> {
        let foo_scalar = egraph[subst[self.foo]].data.scalar;
        let bar_scalar = egraph[subst[self.bar]].data.scalar;
        let new_scalar = foo_scalar + bar_scalar;
        let new_var = egraph.add(TnsrLang::Var(new_scalar.to_string().into()));
        let new_node = egraph.add(TnsrLang::Input([new_var]));
        if egraph.union(new_node, matched_id) {
            vec![new_node, matched_id]
        } else {
            vec![]
        }
    }
}

pub struct TnsrCost<'a> {
    pub egraph: &'a EGraph<TnsrLang, TnsrAnalysis>,
}

impl CostFunction<TnsrLang> for TnsrCost<'_> {
    type Cost = f32;
    /// Getting total cost for the subtree rooted at enode. See egg::CostFunction
    /// trait for more information on interface.
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &TnsrLang, mut costs: C) -> Self::Cost {
        let self_cost = get_op_cost(&*self.egraph, enode);
        enode.fold(self_cost as f32, |sum, id| sum + costs(id))
    }
}

impl LpCostFunction<TnsrLang, TnsrAnalysis> for TnsrCost<'_> {
    fn node_cost(&mut self, _: &EGraph<TnsrLang, TnsrAnalysis>, _: Id, enode: &TnsrLang) -> f64 {
        let self_cost = get_op_cost(&*self.egraph, enode);
        self_cost
    }
}

const COPY_COST: f64 = 0.1;
const ADDITION_COST: f64 = 1.0;
const MULTIPLY_COST: f64 = 6.0;
const DIVISION_COST: f64 = 20.0;
const POW_COST: f64 = 10.0;
const RELU_COST: f64 = 1.0;
const FLOOR_COST: f64 = 0.5;
const MOD_COST: f64 = 0.5;
const MAX_COST: f64 = 0.5;
const MIN_COST: f64 = 0.5;

fn get_op_cost(egraph: &EGraph<TnsrLang, TnsrAnalysis>, enode: &TnsrLang) -> f64 {
    let x = |i: &Id| &egraph[*i].data;
    match enode {
        TnsrLang::Var(_)
        | TnsrLang::Input(_)
        | TnsrLang::Noop(_) => 0.0,

        TnsrLang::Add([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * ADDITION_COST
            }
        }

        TnsrLang::Sub([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * ADDITION_COST
            }
        }

        TnsrLang::Mul([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * MULTIPLY_COST
            }
        }

        TnsrLang::Div([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * DIVISION_COST
            }
        }

        TnsrLang::Smul([s, t]) => {
            assert!(x(s).dtype == DataKind::Scalar);
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(s).constant_foldable && x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f64 * MULTIPLY_COST
            }        
        }

        TnsrLang::MatMul([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = matmul_dimension(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * MULTIPLY_COST
            }
        }

        TnsrLang::Transpose([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f64 * COPY_COST
            }
        }

        TnsrLang::Neg([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr || x(t).dtype == DataKind::Scalar);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f64 * COPY_COST
            }
        }

        TnsrLang::Relu([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f64 * RELU_COST
            }
        }

        TnsrLang::Pow([t, s]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            assert!(x(s).dtype == DataKind::Tnsr);
            if x(t).constant_foldable && x(s).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(t).dims, &x(s).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * POW_COST
            }
        }

        TnsrLang::Floor([t]) => {
            assert!(x(t).dtype == DataKind::Tnsr);
            if x(t).constant_foldable {
                0.0
            } else {
                let product: i32 = x(t).dims.iter().product::<i32>();
                product as f64 * FLOOR_COST
            }
        }

        TnsrLang::Mod([a, b]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let dim = broadcast_vectors(&x(a).dims, &x(b).dims);
                let product: i32 = dim.iter().product::<i32>();
                product as f64 * MOD_COST
            }
        }

        TnsrLang::Max([a, b, c]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            assert!(x(c).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let product: i32 = x(a).dims.iter().product::<i32>();
                product as f64 * MAX_COST
            }
        }

        TnsrLang::Min([a, b, c]) => {
            assert!(x(a).dtype == DataKind::Tnsr);
            assert!(x(b).dtype == DataKind::Tnsr);
            assert!(x(c).dtype == DataKind::Tnsr);
            if x(a).constant_foldable && x(b).constant_foldable {
                0.0
            } else {
                let product: i32 = x(a).dims.iter().product::<i32>();
                product as f64 * MIN_COST
            }
        }

        TnsrLang::FMul([a, b]) => {
            assert!(x(a).dtype == DataKind::Scalar);
            assert!(x(b).dtype == DataKind::Scalar);
            0.0
        }

        TnsrLang::FAdd([a, b]) => {
            assert!(x(a).dtype == DataKind::Scalar);
            assert!(x(b).dtype == DataKind::Scalar);
            0.0
        }
    }
}