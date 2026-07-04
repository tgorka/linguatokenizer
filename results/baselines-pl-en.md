# Baseline Comparison

Target language: `pl` vs. English (FLORES-aligned)

| Baseline | PL fertility | EN fertility | fertility_vs_english | Status |
|----------|-------------:|-------------:|----------------------:|--------|
| gpt2 | 3.713 | 1.230 | 3.018 | scored |
| meta-llama/Llama-3.1-8B | 2.683 | 1.282 | 2.093 | scored |
| mistralai/Mistral-7B-v0.1 | 2.971 | 1.411 | 2.106 | scored |
| Qwen/Qwen2.5-7B | 2.497 | 1.257 | 1.987 | scored |
| deepseek-ai/DeepSeek-V3 | 2.332 | 1.230 | 1.896 | scored |
| speakleash/Bielik-11B-v2.3-Instruct | 2.971 | 1.411 | 2.106 | scored |
| google/gemma-2-9b | -- | -- | -- | skipped: failed to load baseline 'google/gemma-2-9b': You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/google/gemma-2-9b. 403 Client Error. (Request ID: Root=1-6a48bb53-33c253b37f5431951289d193;3778e7e7-bd56-48ee-aada-79602a84d2ff) Cannot access gated repo for url https://huggingface.co/google/gemma-2-9b/resolve/main/config.json. Access to model google/gemma-2-9b is restricted and you are not in the authorized list. Visit https://huggingface.co/google/gemma-2-9b to ask for access. |
| speakleash/Bielik-11B-v3.0-Instruct | -- | -- | -- | skipped: failed to load baseline 'speakleash/Bielik-11B-v3.0-Instruct': You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct. 403 Client Error. (Request ID: Root=1-6a48bb54-77e218243d975454619f2f2a;e21a7941-ccbe-4e4d-b98b-c3cac7f37e22) Cannot access gated repo for url https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct/resolve/main/config.json. Access to model speakleash/Bielik-11B-v3.0-Instruct is restricted and you are not in the authorized list. Visit https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct to ask for access. |
