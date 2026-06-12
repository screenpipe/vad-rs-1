use std::path::Path;

use eyre::Result;

/// ort rc.12: builder ops return `Error<SessionBuilder>` (recovery payload,
/// not Send+Sync) — convert via Display for eyre.
fn oe<R>(e: ort::Error<R>) -> eyre::Report {
    eyre::eyre!("ort: {e}")
}

pub fn create_session<P: AsRef<Path>>(path: P) -> Result<ort::session::Session> {
    let b = ort::session::Session::builder().map_err(oe)?;
    let b = b
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(oe)?;
    let b = b.with_intra_threads(1).map_err(oe)?;
    let mut b = b.with_inter_threads(1).map_err(oe)?;
    let session = b.commit_from_file(path.as_ref()).map_err(oe)?;
    Ok(session)
}
