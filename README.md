# NV HW 4

Implementation of a HiFiGAN


[WanDB Report](https://wandb.ai/kkorolev/tts_project/reports/HW3-TTS--Vmlldzo2MDQ1MTg5)

See the results at the end of this README.

## Checkpoints
- [Model checkpoint 70 epochs](https://disk.yandex.ru/d/qQx-LW21qd17Xg)

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