# Zebrapose4Xray

## Test Environment

- Ubuntu 24.04
- Python 3.8

---

## Workflow

### 1. Mesh Clustering

The point cloud is recursively partitioned using binary clustering.

Both standard **KMeans** and **balanced KMeans** were tested. The final implementation uses standard KMeans.

- Input mesh: `welsh-dragon-small-centered.stl`
- Number of levels: 10
- Output: `binary_level_{level}.vtk` (10 files total)

Each level’s binary labels are stored in the `"clusters"` attribute of the `.vtk` file.

---

### 2. Generate Training Data

```bash
python generate_training_data.py \
    --output_root ./dataset_normal \
    --start_id 0 \
    --num_samples 2000
```

This step:

- samples 2000 random seeds
- generates random camera 6DoF poses
- applies rotation and translation to the mesh
- renders X-ray images
- produces corresponding masks and 10-bit binary codes

### Output Structure

```text
dataset/
├── sample_000000/
│   ├── xray.png
│   ├── mask.png
│   ├── mask.npy
│   ├── code_stack.npy
│   ├── packed_code.npy
│   ├── code_vis.png
│   └── sample_meta.json
├── sample_000001/
│   ├── xray.png
│   ├── mask.png
│   ├── mask.npy
│   ├── code_stack.npy
│   ├── packed_code.npy
│   ├── code_vis.png
│   └── sample_meta.json
...
```

### File Description

- **xray.png**  
  Input image to the neural network.

- **mask.npy** `(H, W)`  
  - `0` = background
  - `1` = object

- **code_stack.npy** `(10, H, W)`  
  - contains 10 binary code channels
  - only valid where `mask == 1`
  - background is all zeros

- **packed_code.npy**  
  Encoded as:

  ```text
  code = b0 + 2b1 + 4b2 + ... + 2^9 b9
  ```

- **code_vis.png**  
  Visualization for quick sanity checking.

---

## Training

### Basic Training

```bash
python train.py \
    --data_root ./dataset_test \
    --save_dir ./checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --backbone resnet34
```

### Full Training Configuration

```bash
python train.py \
    --data_root ./dataset_normal \
    --save_dir ./checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --backbone resnet34 \
    --alpha 1.0 \
    --sigma 3.0 \
    --momentum 0.9 \
    --use_pred_mask_for_code
```

### Training with Pretrained Backbone

```bash
python train_resnet_codebook.py \
    --data_root ./dataset \
    --save_dir ./checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --backbone resnet18 \
    --pretrained
```

### Training Outputs

After training, `save_dir` will contain:

- `history.json` / `history.csv`
- `best_metrics.json` — best epoch and corresponding metrics
- `last.pth` / `best.pth` — model checkpoints

---

## Analyze Training Results

```bash
python analyze_training_results.py \
    --save_dir ./checkpoints \
    --out_dir ./checkpoints/analysis
```

---

## Notes

- Binary codes are only meaningful inside object regions (`mask == 1`).
- Background pixels are always zero.
- `code_vis.png` is intended only for visualization, not training.
