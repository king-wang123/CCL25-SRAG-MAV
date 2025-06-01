import openai

class QwenGen:
    def __init__(self, port=35000, temperature=0):
        """Initialize the QwenGen class with port and temperature settings."""
        self.temperature = temperature
        self.port = port
        self.client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

    def response(self, prompt):
        """Generate a response for the given prompt."""
        tmp_repeat = 0
        while True:
            try:
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                completion = self.client.chat.completions.create(
                    model='default',
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024
                )

                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tmp_repeat += 1
                print(f'repeat {tmp_repeat}')
                if tmp_repeat == 5:
                    break
        return ''

# CUDA_VISIBLE_DEVICES="1,5" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data3/zlh/king/CCL2025-Chinese-Hate-Speech-Detection/models/Qwen2___5-1___5B-Instruct-traindata_train_split_argument_3/full/sft" --trust-remote-code --tensor-parallel-size=2 --port="35000" --max_model_len 10000
