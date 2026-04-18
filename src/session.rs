use std::path::Path;

use eyre::Result;

pub fn create_session<P: AsRef<Path>>(path: P) -> Result<ort::session::Session> {
    let session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(path.as_ref())?;
    Ok(session)
}
