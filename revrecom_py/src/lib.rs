use pyo3::prelude::*;
use std::io::Result as IoResult;
use std::sync::{Arc, Mutex};
use std::collections::BTreeSet;

use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::{RecomParams, RecomVariant, RecomProposal};
use frcw::recom::run::multi_chain;
use frcw::stats::{StatsWriter, SelfLoopCounts};

/// Convert the crate's partition assignments to 0-based labels for Python.
fn to_zero_based(p: &Partition) -> Vec<u32> {
    let max = *p.assignments.iter().max().unwrap_or(&0);
    let need_shift = max >= p.num_dists; // if any equals k, they are 1-based
    if need_shift {
        p.assignments.iter().map(|&a| (a - 1) as u32).collect()
    } else {
        p.assignments.iter().map(|&a| a as u32).collect()
    }
}

/// Writer that records each accepted plan (as 0-based labels).
struct CollectWriter {
    buf: Arc<Mutex<Vec<Vec<u32>>>>,
}
impl CollectWriter {
    fn new(buf: Arc<Mutex<Vec<Vec<u32>>>>) -> Self { Self { buf } }
}
impl StatsWriter for CollectWriter {
    fn init(&mut self, _g: &Graph, p: &Partition) -> IoResult<()> {
        self.buf.lock().unwrap().push(to_zero_based(p));
        Ok(())
    }
    fn step(
        &mut self,
        _step: u64,
        _g: &Graph,
        p: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> IoResult<()> {
        self.buf.lock().unwrap().push(to_zero_based(p));
        Ok(())
    }
    fn close(&mut self) -> IoResult<()> { Ok(()) }
}

// NOTE: edges are 0-based here.
fn edge_list_text(edges: &[(u32,u32)]) -> String {
    let mut s = String::with_capacity(edges.len() * 10);
    for &(u,v) in edges { s.push_str(&format!("{u} {v}\n")); }
    s
}
fn pop_text(pops: &[u32]) -> String {
    pops.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}
fn assign_text(labels_1based: &[u32]) -> String {
    labels_1based.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

#[pyfunction]
#[pyo3(text_signature="(edges, pops, labels, k, num_steps, tol, balance_ub=0, rng_seed=1, n_threads=1, batch_size=1)")]
fn run_revrecom(
    edges: Vec<(u32,u32)>,   // 0-based node ids from Python
    pops: Vec<u32>,
    labels: Vec<u32>,        // 0-based labels from Python
    k: u32,                  // currently unused; kept for API compatibility
    num_steps: u64,
    tol: f64,
    balance_ub: u32,
    rng_seed: u64,
    n_threads: usize,
    batch_size: usize,
) -> PyResult<Vec<Vec<u32>>> {
    // ----- sizes -----
    let n: u32 = pops.len() as u32;
    if labels.len() as u32 != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("labels length {} must equal N={}", labels.len(), n)
        ));
    }

    // ----- EDGES: KEEP 0-BASED (do NOT add +1) -----
    for &(u,v) in &edges {
        if u >= n || v >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("edge ({u},{v}) references node outside 0..={}", n - 1)
            ));
        }
    }

    // ----- LABELS: 1-base ONLY the assignment string for Partition ctor -----
    let labels1: Vec<u32> = labels.iter().map(|&a| a + 1).collect();
    let set1: BTreeSet<u32> = labels1.iter().copied().collect();
    let k_eff: u32 = *set1.iter().max().unwrap();
    let expect: BTreeSet<u32> = (1..=k_eff).collect();
    if set1 != expect {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("assignment 1-based labels mismatch; got {:?}, expected 1..={}", set1, k_eff)
        ));
    }

    // Build graph from 0-based edges
    let g = Graph::from_edge_list(&edge_list_text(&edges), &pop_text(&pops))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("graph error: {e:?}")))?;

    // FAST FAIL if anything in g.edges is >= N (defensive against parsing surprises)
    let n_nodes: usize = g.neighbors.len();
    if let Some(e) = g.edges.iter().find(|e| (e.0 as usize) >= n_nodes || (e.1 as usize) >= n_nodes) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("edge endpoint out of range after Graph build: found ({},{}) with N={n_nodes}. \
                     Edges must be 0-based (0..N-1).", e.0, e.1)
        ));
    }

    let p = Partition::from_assignment_str(&g, &assign_text(&labels1))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("assignment error: {e:?}")))?;

    // ----- 4) Internal sanity: 0..k_eff-1 and no empty districts ------------
    if let (Some(&mn), Some(&mx)) = (p.assignments.iter().min(), p.assignments.iter().max()) {
        if mn < 0 || (mx as u32) >= k_eff {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("partition labels out of range [{mn}..{mx}] for k_eff={k_eff} (expected 0..{})", k_eff - 1)
            ));
        }
    }
    let mut counts = vec![0usize; k_eff as usize];
    for &a in &p.assignments { counts[a as usize] += 1; }
    if let Some((idx,_)) = counts.iter().enumerate().find(|&(_i,&c)| c == 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("empty district detected after construction: id={idx} (0-based) for k_eff={k_eff}")
        ));
    }

    // ----- 5) Params with k_eff ---------------------------------------------
    // (We ignore `k` and use the effective K from labels to avoid mismatches.)
    let avg_pop = (g.total_pop as f64) / (k_eff as f64);

    // bounds from tol (your target)
    let mut min_pop = ((1.0 - tol) * avg_pop).floor() as u32;
    let mut max_pop = ((1.0 + tol) * avg_pop).ceil() as u32;

    // widen to the *observed* seed so first A∪B fits crate buffers
    let obs_min = *p.dist_pops.iter().min().unwrap_or(&0);
    let obs_max = *p.dist_pops.iter().max().unwrap_or(&0);
    if obs_min < min_pop { min_pop = obs_min; }
    if obs_max > max_pop { max_pop = obs_max; }

    // optional: visibility while debugging
    eprintln!(
        "[revrecom_py] K={k_eff} avg_pop={avg_pop:.2} tol={tol:.4}  \
        min_pop={min_pop} max_pop={max_pop}  (obs_min={obs_min} obs_max={obs_max})"
    );

    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps,
        rng_seed,
        balance_ub,
        variant: RecomVariant::Reversible,
        region_weights: None,
    };

    // ----- 6) Collect accepted plans ----------------------------------------
    let buf: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    let writer: Box<dyn StatsWriter> = Box::new(CollectWriter::new(buf.clone()));
    multi_chain(&g, &p, writer, &params, n_threads, batch_size);

    let out = {
        let guard = buf.lock().unwrap();
        guard.clone()
    }; // guard drops here
    Ok(out)
}

#[pyfunction]
fn _wrapper_sig() -> &'static str {
    "revrecom_py wrapper sig: edges=0-based input; labels=0-based → 1-based only for from_assignment_str"
}

#[pymodule]
fn revrecom_py(_py: Python, m: &pyo3::Bound<pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_revrecom, m)?)?;
    m.add_function(wrap_pyfunction!(_wrapper_sig, m)?)?;
    Ok(())
}
