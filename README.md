# NV HW 4

Implementation of a HiFiGAN


[WanDB Report](https://wandb.ai/kkorolev/nv_project/reports/Neural-Vocoder-HW4--Vmlldzo2MTUxMTE0)

See the results at the end of this README.

## Checkpoints
- [Model checkpoint 70 epochs](https://disk.yandex.ru/d/XQg7xYpBUfrl2g)

## Installation guide

```shell
pip install -r ./requirements.txt
```

To reproduce training download LJSpeech.
```shell
sh scripts/download_data.sh
```

Configs can be found in hw_tts/configs folder. In particular, for testing use `config_server.json`.

## Training
One can redefine parameters which are set within config by passing them in terminal via flags.
```shell
python train.py -c CONFIG -r CHECKPOINT -k WANDB_KEY --wandb_run_name WANDB_RUN_NAME --n_gpu NUM_GPU --batch_size BATCH_SIZE --len_epoch ITERS_PER_EPOCH --data_path PATH_TO_WAVS
```

## Testing
```shell
python test.py -c hw_tts/configs/config_server.json -r CHECKPOINT -t test_audio -o output_audio
```
- `test_audio` is a directory with 3 wavs for evaluation.
- `output_audio` is a directory to save the result.

## Results
Generation of these 3 sentences. Filename corresponds to the order of a sentence.

`A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`

`Massachusetts Institute of Technology may be best known for its math, science and engineering education`

`Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`


https://github.com/kkorolev1/nv_dla/assets/72045472/e16cdd34-a78c-4546-bebd-af1d2b547dd5



https://github.com/kkorolev1/nv_dla/assets/72045472/a029865e-f41e-4c7c-8516-6be49df0b4d5



https://github.com/kkorolev1/nv_dla/assets/72045472/ec143fdd-55a3-424e-9152-96bab910b289


