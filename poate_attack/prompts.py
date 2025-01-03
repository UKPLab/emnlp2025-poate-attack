"""
File containing functions to create prompts for different models.
"""
from typing import List, Tuple
import fastchat

from poate_attack.config import ConvTemplates


LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def create_mistral_prompt(template_type: str, prompt: str, system_prompt: str) -> str:
    """Creates a prompt for the model to generate an answer to a question.

    Args:
        prompt (str): The question to be answered.
        system_prompt (str, optional): The system prompt to be used. Defaults to SYSTEM_PROMPT.
    Returns:
        str: The prompt to be sent.
    """
    if template_type=="default":
        if system_prompt:
            return f"[INST] {system_prompt} {prompt} [/INST]"
        else:
            return f"[INST] {prompt} [/INST]"
    elif template_type=="fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("Mistral_7b_instruct")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

def create_llama2_prompt(template_type: str, prompt: str, system_prompt: str):
    if template_type=="default":
        if system_prompt:
            template = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            template = f"<s>[INST] {prompt} [/INST]"
        return template

    elif template_type=="modified":
        if system_prompt:
            template = f"""
<<SYS>>
{system_prompt}
<</SYS>>
[INST]
User:{prompt}
[/INST]
Assistant:
""".strip()
        else:
            template = f"""
[INST]
User:{prompt}
[/INST]
Assistant:
""".strip()
        return template
    elif template_type=="fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("Llama_2_7b_chat_hf")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        template = "<s>"+conv.get_prompt()  # solve issue for VLLM inference by adding <s> token

        return template

def create_gpt_prompt(template_type: str, prompt: str, system_prompt: str):

    if template_type=="default":
        if system_prompt:
            template = [("system", system_prompt), ("human", prompt)]
        else:
            template = [("human", prompt)]
    elif template_type=="fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("gpt-35-turbo")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        template = conv.get_prompt()
    return template


def create_llama3_prompt(template_type: str, system_prompt: str, prompt: str, output:str=None):
    if template_type=="default":
        if system_prompt:
            template = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
        else:
            template = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
        return template

    elif template_type=="modified":
        if system_prompt:
            template = f"""
<|begin_of_text|>
<<system>>
{system_prompt}
<</system>>
[INST]
user:{prompt}
[/INST]
assistant:
""".strip()
        else:
            template = f"""
<|begin_of_text|>
<|start_header_id|>
user:{prompt}
<|end_header_id|>
assistant:
""".strip()
        return template

    elif template_type == "fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("Llama_3_8b_instruct")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], output)
        template = conv.get_prompt()
        return template


def create_llama3_few_shot_prompt(system_prompt: str, examples: List[Tuple[str, str]], prompt: str):

    conv_templates = ConvTemplates()
    conv = conv_templates.get_template("Llama_3_8b_instruct")
    conv.system_message = system_prompt if system_prompt else ""

    for prompt, output in examples:
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], output)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    template = conv.get_prompt()
    return template


def create_falcon_prompt(template_type: str, prompt: str, system_prompt: str, output:str=None):
    if template_type == "default":
        if system_prompt:
            template = ""
        return template

    elif template_type == "fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("falcon_7b_instruct")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], output)
        template = conv.get_prompt()
        return template

def create_gemma_prompt(template_type: str, system_prompt: str, prompt: str, output:str=None):
    if template_type=="default":
        if system_prompt:
            template = ""
        return template

    elif template_type == "fastchat":
        conv_templates = ConvTemplates()
        conv = conv_templates.get_template("gemma2_2b_it")
        conv.system_message = system_prompt if system_prompt else ""
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], output)
        template = conv.get_prompt()
        return template


def create_phi_prompt(template_type: str, prompt: str, system_prompt: str):
    # ignoring template type here
    if system_prompt:
        template = f"""
<|system|>
{system_prompt}<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>
""".strip()

    else:
        template = f"""
<|user|>
{prompt}<|end|>
<|assistant|>
""".strip()
    return template


def load_conversation_template(model_name: str):
    """
    load the chat template function for the target model
    :return:
    """
    model_path_lower = model_name.lower()
    if any(keyword in model_path_lower for keyword in ["llama3", "llama_3", "llama_3.1", "llama3.1"]):
        conv = fastchat.model.get_conversation_template("llama-3")
    elif any(keyword in model_path_lower for keyword in ["llama2", "llama-2", "llama_2"]):
        conv = fastchat.model.get_conversation_template("llama-2")
        conv.sep2 = conv.sep2.strip()
    elif any(keyword in model_path_lower for keyword in ["vicuna"]):
        conv = fastchat.model.get_conversation_template("vicuna")
    elif any(keyword in model_path_lower for keyword in ["mistral", "mistralai"]):
        conv = fastchat.model.get_conversation_template("mistral")
    elif any(keyword in model_path_lower for keyword in ["phi"]):
        conv = create_phi_prompt
    elif any(keyword in model_path_lower for keyword in ["gemma"]):
        conv = fastchat.model.get_conversation_template("gemma")
    elif any(keyword in model_path_lower for keyword in ["gpt"]):
        conv = fastchat.model.get_conversation_template("chatgpt")
    else:
        raise ValueError(f"Model {model_name} not supported for inference")

    return conv


# Test the functions
if __name__ == '__main__':
    prompt = create_llama2_prompt(
        template_type="fastchat", prompt="What is the capital of France?", system_prompt="please answer the question."
    )
    print(prompt)
