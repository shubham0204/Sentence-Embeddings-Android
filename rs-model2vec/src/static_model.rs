use anyhow::Ok;
use anyhow::Result;
use memmap2::MmapOptions;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use safetensors::SafeTensors;
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub struct StaticModel {
    tokenizer: Tokenizer,
    pub embedding_dims: usize,
    embeddings_u8: Vec<u8>,
    seq_buffer: Vec<String>,
}

impl StaticModel {
    pub fn new(
        embeddings_filepath: &str,
        tokenizer_filepath: &str,
        num_threads: usize,
    ) -> Result<Self> {
        // set num threads for Rayon
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()?;
        // load the tokenizer
        let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_filepath).expect("");
        // load the embeddings
        let tensor_file: File = File::open(embeddings_filepath)?;
        let buffer = unsafe { MmapOptions::new().map(&tensor_file)? };
        let tensors = SafeTensors::deserialize(&buffer)?;
        let embeddings_tensor_view = tensors.tensor("embeddings")?;
        let embeddings_u8 = embeddings_tensor_view.data().to_vec();
        let embedding_dims = embeddings_tensor_view.shape()[1];
        Ok(StaticModel {
            tokenizer,
            embedding_dims,
            embeddings_u8,
            seq_buffer: Vec::new(),
        })
    }

    pub fn add_seq_buffer(&mut self, sequence: String) {
        self.seq_buffer.push(sequence);
    }

    pub fn clear_seq_buffer(&mut self) {
        self.seq_buffer.clear();
    }

    pub fn encode_seq_buffer(&self) -> Result<Vec<Vec<f32>>> {
        self.encode(&self.seq_buffer)
    }

    pub fn encode(&self, sequences: &Vec<String>) -> Result<Vec<Vec<f32>>> {
        // tokenize the input sequences
        let tokenized_sequences = self
            .tokenizer
            .encode_batch(sequences.to_vec(), false)
            .expect("tokenizer.encode_batch failed");

        // use a mutex to ensure atomic access to the embeddings Vec.
        let embeddings_mutex: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));

        tokenized_sequences.par_iter().for_each(|sequence| {
            let ids: &[u32] = sequence.get_ids();

            // the sentence embeddings, obtained after pooling/averaging
            // all `token_embedding`
            let mut sentence_embedding: Vec<f32> = vec![0.0; self.embedding_dims];
            for id in ids {
                // for each id in the tokenized sequence
                let token_embedding: &[f32] = self.get_embedding(*id as usize);
                for di in 0..self.embedding_dims {
                    sentence_embedding[di] += token_embedding[di];
                }
            }
            for di in 0..self.embedding_dims {
                sentence_embedding[di] /= ids.len() as f32;
            }
            embeddings_mutex.lock().unwrap().push(sentence_embedding);
        });

        let embeddings = embeddings_mutex.lock().unwrap().clone();
        Ok(embeddings)
    }

    fn get_embedding(&self, index: usize) -> &[f32] {
        // slice the raw embeddings data
        let embedding_raw: &[u8] = &self.embeddings_u8
            [index * self.embedding_dims * 4..(index + 1) * self.embedding_dims * 4];

        // cast embedding_raw to a *const f32 to parse as a float-array
        // instead of u8 array
        let len: usize = embedding_raw.len();
        let ptr: *const f32 = embedding_raw.as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len / 4) }
    }
}

#[cfg(test)]
mod test_static_model {
    use super::StaticModel;

    #[test]
    fn test_new() {
        let static_model = StaticModel::new("embeddings.safetensors", "tokenizer.json", 2).unwrap();
        assert!(static_model.embedding_dims == 256);
        assert!(static_model.tokenizer.get_vocab_size(true) == 29528);
        assert!(static_model.embeddings_u8.len() > 0);
    }

    #[test]
    fn test_encoding() {
        let static_model = StaticModel::new("embeddings.safetensors", "tokenizer.json", 2).unwrap();
        let sequences = vec![String::from("Hello World")];
        let embeddings = static_model.encode(&sequences).unwrap();
        assert!(embeddings.len() == 1);
        assert!(embeddings[0].len() == 256);
    }
}