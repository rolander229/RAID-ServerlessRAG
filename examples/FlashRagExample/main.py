from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
import argparse

def main(args):
    config_dict = {
                    'data_dir': 'dataset/',
                    'index_path': 'indexes/e5_Flat.index',
                    'corpus_path': 'indexes/general_knowledge.jsonl',
                    'model2path': {'e5': args.retrieval_path, 'Qwen2-0.5B': args.generator_path},
                    'generator_model': 'Qwen2-0.5B',
                    'retrieval_method': 'e5',
                    'metrics': ['em','f1','acc'],
                    'retrieval_topk': 1,
                    'save_intermediate_data': True
                }

    config = Config(config_dict = config_dict)

    all_split = get_dataset(config)
    test_data = all_split['test']
    pipeline = SequentialPipeline(config)

    output_dataset = pipeline.run(test_data,do_eval=True)
    print("---generation output---")
    print(output_dataset.pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_path', type=str, required=False)
    parser.add_argument('--generator_path', type=str, required=False)
    args = parser.parse_args()
    args.retrieval_path = './e5-base-v2'
    args.generator_path = './Qwen3-0.6B'
    main(args)
