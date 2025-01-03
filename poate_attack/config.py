import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from fastchat.model import get_conversation_template

from dotenv import load_dotenv

load_dotenv()

BASE_PATH = os.getenv('BASE_PATH')

@dataclass
class ModelPath:
    # Class variable to hold the paths
    paths: ClassVar[Dict[str, str]] = {
        "Llama_2_7b_chat_hf": f"{BASE_PATH}models/llama2_7b_chat_hf",
        "Llama_2_70b_chat_hf": f"meta-llama/Llama-2-70b-chat-hf",
        "Llama_2_13b_chat_hf": "meta-llama/Llama-2-13b-chat-hf",
        "Llama_3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",  #
        "Llama_3_8b": "meta-llama/Meta-Llama-3-8B",
        "Llama_2_7b_chat_sd": f"{BASE_PATH}/models/llama2_safe_decoding/final_checkpoint",
        "vicuna_7b": f"{BASE_PATH}/models/vicuna_7b",
        "vicuna_13b": f"{BASE_PATH}/models/vicuna_13b",
        "phi_3_mini_4k": "microsoft/Phi-3-mini-4k-instruct",
        "phi_3_small_8k": "microsoft/Phi-3-small-8k-instruct",
        "phi_3_medium_4k": "microsoft/Phi-3-medium-4k-instruct",
        "Mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "Llama_3.1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama_3.1_70b_instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "gemma2_2b_it": "google/gemma-2-2b-it",
        "gemma2_2b": "google/gemma-2-2b",
        "gemma2_9b_it": "google/gemma-2-9b-it",
        "gemma2_27b_it": "google/gemma-2-27b-it",
        "gemma2_9b": "google/gemma-2-9b",
        "Llama_3.1_8b_potee": "Llama-3.1-8b-instruct-hf-jailbreak/potee_attack_safe_question_2024-08-19/final_checkpoint/",
        "falcon_7b_instruct": f"{BASE_PATH}/models/falcon_7b_instruct",
    }

    @classmethod
    def get_path(cls, model_name: str) -> str:
        return cls.paths.get(model_name, "Path not found")

@dataclass
class ConvTemplates:

    template_names: ClassVar[Dict[str, str]] = {
        "Llama_2_7b_chat_hf": "llama-2",
        "Llama_2_13b_chat_hf": "llama-2",
        "Llama_2_70b_chat_hf": "llama-2",
        "Llama_2_7b_chat_sd": "llama-2",
        "vicuna_7b": "vicuna",
        "vicuna_13b": "vicuna",
        "Llama_3_8b_instruct": "llama-3",
        "Llama_3_8b": "llama-3",
        "Llama_3.1_8b_instruct": "llama-3",
        "Llama_3.1_70b_instruct": "llama-3",
        "Mistral_7b_instruct": "mistral",
        "gemma2_2b_it": "gemma",
        "gemma2_2b": "gemma",
        "gemma2_9b_it": "gemma",
        "gemma2_9b": "gemma",
        "gemma2_27b_it": "gemma",
        "Llama_3.1_8b_potee": "llama-3",
        "falcon_7b_instruct": "falcon",
        "gpt-35-turbo": "chatgpt",
    }

    def get_template_name(self, model_name: str) -> str:
        return self.template_names.get(model_name, "Template not found")

    def get_template(self, model_name: str):
        template = get_conversation_template(self.get_template_name(model_name))
        if template.name == "llama-2":
            template.sep2 = template.sep2.strip()
        return template
