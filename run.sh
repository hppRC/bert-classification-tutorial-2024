# accelerate launch --config_file config/4-ds.json src/train.py --model_name cl-tohoku/bert-base-japanese-v2
# accelerate launch --config_file config/4-ds.json src/train.py --model_name studio-ousia/luke-japanese-base-lite
# accelerate launch --config_file config/4-ds.json src/train.py --model_name tohoku-nlp/bert-base-japanese-v3
# accelerate launch --config_file config/4-ds.json src/train.py --model_name studio-ousia/luke-japanese-large-lite
# accelerate launch --config_file config/4-ds.json src/train.py --model_name cl-tohoku/bert-large-japanese-v2
accelerate launch --config_file config/4-ds.json src/train.py --model_name tokyotech-llm/Swallow-MS-7b-v0.1 --experiment_name clip-grad
accelerate launch --config_file config/4-ds.json src/train.py --model_name tokyotech-llm/Swallow-MS-7b-instruct-v0.1 --experiment_name clip-grad
accelerate launch --config_file config/4-ds.json src/train.py --model_name mistralai/Mistral-7B-Instruct-v0.1 --experiment_name clip-grad
accelerate launch --config_file config/4-ds.json src/train.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --experiment_name clip-grad