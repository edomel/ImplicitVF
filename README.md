# ImplicitVF: Neural Vector Fields for Implicit Surface Representation and Inference

[[Arxiv]](https://arxiv.org/abs/2204.06552)

![method_fig](figs/method.png)

The code, which implements the paper "Neural Vector Fields for Implicit Surface Representation and Inference", uses the Vector Field (VF) as an implicit representation for 3D surfaces.
Example reconstructions with the proposed method can be seen inside [meshes](meshes/).

## Citing ImplicitVF

If you use ImplicitVF in your research, please cite the
[paper](https://arxiv.org/abs/2204.06552):
```
@misc{mellorella2023neural,
      title={Neural Vector Fields for Implicit Surface Representation and Inference}, 
      author={Edoardo Mello Rella and Ajad Chhatkuli and Ender Konukoglu and Luc Van Gool},
      year={2023},
      eprint={2204.06552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## How to Use ImplicitVF

<!-- ### Python Requirements -->

### Dataset

The approach is general and can be applied to any shape for which the meshes are provided.

We provide an implementation tested on the [ShapeNetCoreV2](https://shapenet.org/) dataset. Its extension to other datasets might require minor changes in the layout of the files in the accessed data.

### Create Data Splits

Generates the split file used during training and evaluation

```
python create_split_file.py -d <dataset directory> -c <object class>
```

### Data Preprocessing

Samples VF around the shapes specified in a split file and saves the samples for each shape in a separate file

```
python prepare_data.py -d <target dataset directory> -s <source dataset directory> --split <split filename>
```

### Training

Trains an autodecoder to reconstruct the shapes in a class using VF as the implicit representation.

```
python train.py -e <experiment directory>
```

The other parameters regarding the experiment are specified in the ```specs.json``` files.

### Reconstruction

Uses a trained model to reconstruct the meshes in the evaluation set.

```
python reconstruct.py -e <experiment directory> -c <checkpoint epoch>
```