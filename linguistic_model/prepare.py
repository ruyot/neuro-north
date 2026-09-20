"""Download GPT-2 once; the live UI loads it offline afterward."""
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    destination = Path(__file__).parent / 'models' / 'gpt2'
    snapshot_download('openai-community/gpt2', local_dir=str(destination),
                      allow_patterns=['config.json', 'generation_config.json', 'tokenizer*.json',
                                      'vocab.json', 'merges.txt', 'model.safetensors'])
    print(f'Local GPT-2 ready: {destination}')


if __name__ == '__main__':
    main()
