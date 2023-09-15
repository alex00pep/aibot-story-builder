from langchain import LLMChain, PromptTemplate
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from app.model_loader import load_model

load_dotenv()


# Image to text model
def img2test(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base"
    )
    text = image_to_text(url)[0]["generated_text"]

    return text


import torch

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# Using LLM
def generate_story(scenario: str):
    template = """
    You are astory teller. You can generate ashort story based on the given narrative,
    the story should not be more than 20 words.

    CONTEXT: {scenario}
    STORY:
    """
    llm = load_model(
        device_type=DEVICE_TYPE,
        model_id="TheBloke/Llama-2-7B-Chat-GGML",
        model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin",
        temperature=1.0,
    )
    # llm = pipeline(task="text-generation", model="roberta-large-mnli")

    return llm(template.format(scenario=scenario))[0]["generated_text"]
    # prompt = PromptTemplate(template=template, input_variables=["scenario"])
    # llm.predict(prompt=template.format(scenario=scenario))

    # story_llm = LLMChain(llm=llm, prompt=prompt)


generate_story("Once upon a time ")
