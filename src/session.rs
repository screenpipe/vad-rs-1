use std::path::Path;

use eyre::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};

pub fn create_session<P: AsRef<Path>>(path: P) -> Result<Session> {
    let session = Session::builder()
        .map_err(|e| eyre::eyre!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| eyre::eyre!("{e}"))?
        .with_intra_threads(1)
        .map_err(|e| eyre::eyre!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| eyre::eyre!("{e}"))?
        .commit_from_file(path.as_ref())
        .map_err(|e| eyre::eyre!("{e}"))?;
    Ok(session)
}
