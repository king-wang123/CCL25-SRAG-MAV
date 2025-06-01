from tools import get_json, output2triple, process_triple, check_response
from qwen_gen import QwenGen
from get_prompt import generate_rag_prompt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def retriever(texts, query, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)

    # 将 PyTorch tensor 转换为 NumPy array，如果它们在 GPU 上，则移至 CPU
    if query_embedding.is_cuda:
        query_embedding = query_embedding.cpu()

    query_embedding_np = query_embedding.numpy().reshape(1, -1) # Reshape for cosine_similarity

    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding_np, corpus_embeddings_np)[0]

    # 获取 top_k 个最相似的文本的索引
    if len(similarities) <= top_k:
        # 如果文本数量少于或等于 top_k，则返回所有文本（按相似度降序）
        top_k_indices = np.argsort(similarities)[::-1]
    else:
        # 获取相似度最高的 top_k 个索引
        top_k_indices = np.argsort(similarities)[-top_k:][::-1] # 从大到小排序

    # 返回最相关的文本
    retrieved_texts = [texts[idx] for idx in top_k_indices]
    
    return retrieved_texts


data = get_json('../data/train.json')
texts = [item['content'] for item in data]
test2item = {}
for item in data:
    item['output'] = output2triple(item['output'])
    test2item[item['content']] = item

model_path="../models/bge-large-zh-v1.5"
print(f"正在从本地路径加载模型: {model_path}")
model = SentenceTransformer(model_path)
print("模型加载成功。")
corpus_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
if corpus_embeddings.is_cuda:
    corpus_embeddings = corpus_embeddings.cpu()
corpus_embeddings_np = corpus_embeddings.numpy()

qwen = QwenGen(port=35000, temperature=0.1)
integration_num = 10
threshold = 30

def gen_output(item):
    all_responses = []

    def process_retriever_content(retriver_content):
        retriver_item = test2item[retriver_content]
        prompt = generate_rag_prompt(item['content'], retriver_item, 'triple')
        return qwen.response(prompt)
    
    while True:
        retriver_contents = retriever(texts, item['content'], top_k=integration_num)
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_retriever_content, retriver_contents)
            all_responses.extend(results)

        response_counts = Counter(all_responses)
        most_common_list = response_counts.most_common(1)
        actual_most_common_response = most_common_list[0][0]
        count_of_most_common = most_common_list[0][1]

        if count_of_most_common >= threshold:
            return actual_most_common_response


test_data = get_json('../data/test2.json')
with open(f'../data/output/qwen2_7b_instruct_train_rag_triple_accu_integration{integration_num}_{threshold}_t_0.1.txt', 'w', encoding='utf-8') as f:
    for item in tqdm(test_data):
        item['output'] = process_triple(gen_output(item))

        while not check_response(item['output']):
            item['output'] = process_triple(gen_output(item))
            print(item['output'])
        f.write(item['output'] + '\n')



# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data3/zlh/king/CCL2025-final/models/Qwen2___5-7B-Instruct-traindata_train_rag_triple/full/sft" --trust-remote-code --tensor-parallel-size=4 --port="35000" --max_model_len 10000