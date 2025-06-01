from tools import get_json, save_json, output2triple
from get_prompt import generate_rag_prompt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

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

train_data = []
for item in tqdm(data):
    retriver_content = retriever(texts, item['content'], top_k=2)[1]
    retriver_item = test2item[retriver_content]
    train_data.append({
        'instruction': '',
        'input' : generate_rag_prompt(item['content'], retriver_item, 'triple'),
        'output' : item['output']
    })

save_json(train_data, '../data/train_rag_triple.json')