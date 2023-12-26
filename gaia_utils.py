import tiktoken

INDEX = ''
INDEX_ID = ''
FILE_ID = ''
NODES_ID = []
DOC_ID = []
UPLOADER_FILE = ''
ST = ''
FILE_UPLOAD_KEY = 0
UserInput = ''

MyModel = 0
MyModelHistory = {}
MyModelHistoryMsg = {}
MODEL_AURA = 1
MODEL_GAIA = 2
MODEL_Llama2 = 3
MODEL_WizardCoder = 4

VectorSelected = None

OnesGreetingFlag = {}

def llama_v2_prompt(
    messages: list[dict]
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    # messages_list = messages_list[:-1]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


# if __name__ == "__main__":
#     print(llama_v2_prompt([{"role": "user", "content": "hello"},
#                            {"role": "assitant", "content": "how can i help you today?"},
#                            {"role": "user", "content": "tell me where is tel aviv?"},
#                            {"role": "assitant", "content": "it is in israel. any other questions?"},
#                            {"role": "user", "content": "No. thanks"}]))
#
#     print(llama_v2_prompt([{'role': 'user', 'content': 'how much is two plus two?'}, {'role': 'assistant', 'content': "  Thank you for asking! *adjusts glasses* Two plus two is equal to... (checking notes) ...four! 😊 Yes, that's correct! Two plus two is equal to four. I'm glad I could help. Is there anything else I can assist you with? 😊</s>"}, {'role': 'user', 'content': 'what about 2 + 3?'}, {'role': 'assistant', 'content': '  Sure! 2 + 3 = 5. Unterscheidung is correct.</s>'}, {'role': 'user', 'content': 'how about 3 + 3?'}]))
#
#     print(num_tokens_from_string("Hello world, let's test tiktoken.", "cl100k_base"))
