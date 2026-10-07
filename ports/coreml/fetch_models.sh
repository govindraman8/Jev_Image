#!/bin/bash
# Re-download the pinned prebuilt FluidInference Core ML buckets into ./models and verify SHA-256.
# (typed-decisions has no published build: run `.venv/bin/python convert.py --ckpt typed-decisions --length 128`.)
set -euo pipefail
cd "$(dirname "$0")"
fetch() {  # repo revision path destdir
  mkdir -p "$(dirname "$4/$3")"
  [ -s "$4/$3" ] || curl -sSfL --retry 3 -o "$4/$3" "https://huggingface.co/$1/resolve/$2/$3"
}
ML=FluidInference/laya-coreml; ML_REV=7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc
EN=FluidInference/laya-english-coreml; EN_REV=78c0b0e5054eb5804c72080016227d4f3b0bd08d
for f in README.md config.json NOTICE.md LICENSE; do fetch $ML $ML_REV $f models/multilingual; done
for L in 128 256; do for f in analytics/coremldata.bin coremldata.bin model.mil weights/weight.bin; do
  fetch $ML $ML_REV laya_multilingual_fp16_L${L}_options32.mlmodelc/$f models/multilingual; done; done
for f in README.md CONFIG_PROVENANCE.md LICENSE config.json rl_agent_config.json assets.lock.json calibration.py \
         preprocessing.py laya_english_fp16_L128_options32.conversion.json verification-english-L128.json \
         laya_english_fp16_L128_options32.mlpackage/Manifest.json \
         laya_english_fp16_L128_options32.mlpackage/Data/com.apple.CoreML/model.mlmodel \
         laya_english_fp16_L128_options32.mlpackage/Data/com.apple.CoreML/weights/weight.bin; do
  fetch $EN $EN_REV $f models/english; done
shasum -a 256 -c <<'SUMS'
506eb53e74aaebcdfb3f55304fbdd0b9c1678026efaef60454a1c66557f35d90  models/multilingual/laya_multilingual_fp16_L128_options32.mlmodelc/weights/weight.bin
1a83b587eb200359c041103da261eca89a9b04eda586f41958143479bc975824  models/multilingual/laya_multilingual_fp16_L128_options32.mlmodelc/coremldata.bin
0fbc069ce5b66d0f2a28387c90116c30f95f8555600ccf46d612590edf646e49  models/multilingual/laya_multilingual_fp16_L128_options32.mlmodelc/model.mil
41e287a96c4260faec5c1d2aa37209666279a69541652ea450879e5853881d73  models/multilingual/laya_multilingual_fp16_L256_options32.mlmodelc/weights/weight.bin
8752656ef300dc8b47199da1241ab1ef84cc8d11fd4b5fbe8981f374c09a016f  models/multilingual/laya_multilingual_fp16_L256_options32.mlmodelc/coremldata.bin
d5d79d89a328c0d08ad42bb84c46de013774bde7bdead7aa191477605b238b03  models/multilingual/laya_multilingual_fp16_L256_options32.mlmodelc/model.mil
0a2b893b40aea33eb60223a3cb3dc65cc018d3d9a1d747397552e5043575388a  models/english/laya_english_fp16_L128_options32.mlpackage/Data/com.apple.CoreML/model.mlmodel
dd7329ad7eb98b26ee95ea5e02365ba30da76df563ecaf618faee9e5ad5e6410  models/english/laya_english_fp16_L128_options32.mlpackage/Data/com.apple.CoreML/weights/weight.bin
96a32093f5f1f65d535964068962f1ae489fe863c4b045f0910d44e654503cb4  models/english/laya_english_fp16_L128_options32.mlpackage/Manifest.json
SUMS
