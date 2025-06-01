def generate_prompt(content, type):
    if type == 'triple':
        return f"你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:\n{content}"


def generate_rag_prompt(content, retriver_item, type):
    if type == 'triple':
        return f'你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组:\n示例：\n### 句子：\n{retriver_item["content"]}\n### 三元组：\n{retriver_item["output"]}\n### 句子：\n{content}\n### 三元组：\n'
