use eyre::{bail, Result};
use ndarray::{Array1, Array2, ArrayD};
use std::path::Path;

use crate::{session, vad_result::VadResult};

/// Silero VAD v5/v6 compatible Voice Activity Detector
///
/// Key differences from v4:
/// - State shape: [2, 1, 128] (vs v4's [2, 1, 64])
/// - Context prepended to input (64 samples for 16kHz, 32 for 8kHz)
/// - Outputs: "stateN", "output"
#[derive(Debug)]
pub struct Vad {
    session: ort::session::Session,
    sample_rate: i64,
    state: ArrayD<f32>,
    context: Array1<f32>,
    context_size: usize,
}

impl Vad {
    pub fn new<P: AsRef<Path>>(model_path: P, sample_rate: usize) -> Result<Self> {
        if ![8000_usize, 16000].contains(&sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }

        let session = session::create_session(model_path)?;

        // Silero v5/v6 uses [2, 1, 128] state shape
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());

        // Context size depends on sample rate
        let context_size = if sample_rate == 16000 { 64 } else { 32 };
        let context = Array1::<f32>::zeros(context_size);

        Ok(Self {
            session,
            sample_rate: sample_rate as i64,
            state,
            context,
            context_size,
        })
    }

    pub fn compute(&mut self, samples: &[f32]) -> Result<VadResult> {
        // Concatenate context with input (v5/v6 requirement)
        let mut input_with_context = Vec::with_capacity(self.context_size + samples.len());
        input_with_context.extend_from_slice(self.context.as_slice().unwrap());
        input_with_context.extend_from_slice(samples);

        let frame = Array2::<f32>::from_shape_vec(
            [1, input_with_context.len()],
            input_with_context
        )?;

        // Sample rate as 1D array
        let sr = Array1::from_vec(vec![self.sample_rate]);

        // Run inference with named inputs for v5/v6 format
        // The model expects: input, state, sr
        let result = self.session.run(ort::inputs![
            "input" => ort::value::TensorRef::from_array_view(frame.view())?,
            "state" => ort::value::TensorRef::from_array_view(self.state.view())?,
            "sr" => ort::value::TensorRef::from_array_view(sr.view())?
        ])?;

        // Extract and update state from "stateN" output
        let state_output = result
            .get("stateN")
            .ok_or_else(|| eyre::eyre!("Missing stateN output"))?
            .try_extract_array::<f32>()?;

        // Clone the state data
        let state_slice: Vec<f32> = state_output.iter().copied().collect();
        self.state = ArrayD::from_shape_vec([2, 1, 128].as_slice(), state_slice)?;

        // Update context with last context_size samples from input
        if samples.len() >= self.context_size {
            self.context = Array1::from_vec(
                samples[samples.len() - self.context_size..].to_vec()
            );
        } else {
            // If samples is smaller than context, shift context and append samples
            let mut new_context = Vec::with_capacity(self.context_size);
            new_context.extend_from_slice(&self.context.as_slice().unwrap()[samples.len()..]);
            new_context.extend_from_slice(samples);
            self.context = Array1::from_vec(new_context);
        }

        // Extract probability from "output"
        let output = result
            .get("output")
            .ok_or_else(|| eyre::eyre!("Missing output"))?
            .try_extract_array::<f32>()?;

        let prob = *output.first().ok_or_else(|| eyre::eyre!("Empty output tensor"))?;

        Ok(VadResult { prob })
    }

    pub fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.context = Array1::<f32>::zeros(self.context_size);
    }
}
