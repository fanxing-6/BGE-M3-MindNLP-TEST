
import os
import sys
import faiss
import datasets
import numpy as np
from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from mindspore import ops
from mindnlp.transformers import AutoModel, AutoTokenizer
from mindspore import context
import logging

sys.path.append("..")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def norm(tensor):
    """对生成的嵌入进行归一化"""
    norm_ = ops.L2Normalize(axis=-1, epsilon=1e-12)
    return norm_(tensor)


@dataclass
class ModelArgs:
    encoder: str = field(
        default="liuyanyi/bge-m3-hf",  # 替换为MindSpore模型的路径
        metadata={'help': 'Name or path of encoder'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={
            'help': 'Dir to save index. Corpus index will be saved to `index_save_dir/{encoder_name}/index`. Corpus ids will be saved to `index_save_dir/{encoder_name}/docid` .'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=82,
        metadata={'help': 'Inference batch size.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def get_model(model_args: ModelArgs):
    tokenizer = AutoTokenizer.from_pretrained(model_args.encoder)
    model = AutoModel.from_pretrained(model_args.encoder)
    # model.jit()
    return model, tokenizer


from tqdm import tqdm
import itertools


def parse_corpus(corpus: datasets.Dataset, max_samples=None):
    corpus_list = []

    if max_samples is not None:
        iterator = itertools.islice(corpus, max_samples)
    else:
        iterator = corpus

    # 遍历语料数据并进行处理
    for data in tqdm(iterator, desc="Generating corpus"):
        _id = str(data['_id'])
        content = f"{data['title']}\n{data['text']}".lower()
        content = normalize(content)
        corpus_list.append({"id": _id, "content": content})

    # 将处理后的数据转换为 datasets.Dataset 格式
    processed_corpus = datasets.Dataset.from_list(corpus_list)

    return processed_corpus


import time


def bgeModel(model, inputs):
    outputs = model(**inputs, return_dict=True)
    return outputs


def generate_embeddings(model, tokenizer, texts, max_passage_length=512, batch_size=1000):
    all_embeddings = []
    # 初始化 tqdm 进度条
    pbar = tqdm(total=len(texts), desc="Generating Embeddings")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # 开始计时
        start_time = time.time()
        # 生成嵌入
        inputs = tokenizer(batch_texts, return_tensors="ms", padding=True, truncation=True,
                           max_length=max_passage_length)
        # outputs = model(**inputs, return_dict=True)
        outputs = bgeModel(model, inputs)
        dense_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出
        dense_output = norm(dense_output)  # 归一化嵌入
        all_embeddings.append(dense_output.asnumpy())  # 转换为 numpy 以便后续使用
        end_time = time.time()
        batch_time = end_time - start_time
        # 更新进度条
        pbar.update(len(batch_texts))
        pbar.set_postfix(batch_time=f"{batch_time:.4f} s", batch_size=f"{len(batch_texts)}")

    pbar.close()  # 完成后关闭进度条
    return np.vstack(all_embeddings)


def generate_index(model, tokenizer, corpus: datasets.Dataset, max_passage_length: int = 512, batch_size: int = 512):
    """生成FAISS索引"""
    corpus_embeddings = generate_embeddings(model, tokenizer, corpus["content"], max_passage_length, batch_size)
    dim = corpus_embeddings.shape[-1]

    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])


def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    """保存索引和文档ID"""
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


def main():
    """主流程"""
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs

    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]

    model, tokenizer = get_model(model_args=model_args)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)

    print("==================================================")
    print("Start generating embedding with model:")
    print(model_args.encoder)

    index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder))
    print("index_save_dir", index_save_dir)
    if not os.path.exists(index_save_dir):
        os.makedirs(index_save_dir)
    if os.path.exists(os.path.join(index_save_dir, 'index')) and not eval_args.overwrite:
        print(f'Embedding already exists. Skip...')
        return

    # 加载数据集
    corpus = \
        datasets.load_dataset("/home/ma-user/work/workplace/mindnlp/FlagEmbedding/C_MTEB/MKQA/dense_retrieval/nq",
                              'corpus',
                              trust_remote_code=True)['corpus']
    corpus = parse_corpus(corpus=corpus)

    # 生成索引
    index, docid = generate_index(
        model=model,
        tokenizer=tokenizer,
        corpus=corpus,
        max_passage_length=eval_args.max_passage_length,
        batch_size=eval_args.batch_size
    )

    # 保存结果
    save_result(index, docid, index_save_dir)

    print("==================================================")
    print("Finish generating embeddings with following model:")
    pprint(model_args.encoder)


if __name__ == "__main__":
    main()
