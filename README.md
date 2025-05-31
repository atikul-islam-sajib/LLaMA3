# LLaMA3

This is a simple implementation of the LLaMA3 transformer language model using PyTorch.  
It is trained in an autoregressive fashion to predict the next token in a sequence.

---

## Configuration

### Artifacts

- Model checkpoints and files are saved to:
  - `./artifacts/files`
  - `./artifacts/checkpoints/train_models`
  - `./artifacts/checkpoints/best_model`

### Model (LLaMA3)

- `dimension`: 512  
- `num_vocabularies`: 4096  
- `query_heads`: 8  
- `num_layers`: 16  
- `kv_heads`: 4  
- `eps`: 1e-4  
- `sequence_length`: 128  
- `base`: 10000  
- `output_dimension`: 14336  

### Trainer

- `epochs`: 200  
- `lr`: 1e-5  
- `beta1`: 0.9  
- `beta2`: 0.999  
- `device`: "cuda"  

---

## Files

- `model.py` — defines the LLaMA3 transformer model
- `trainer.py` — training loop using synthetic tokenized input - make sure you have created the dataloader which is appropriated with LLaMA 

---

## How to Run

```bash
python src/trainer.py
````
