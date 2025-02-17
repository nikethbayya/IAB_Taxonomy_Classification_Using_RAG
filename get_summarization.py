import os
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_openai import OpenAI
import pandas as pd
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key="",
             temperature=0.7, max_tokens=100 )


# Function to generate a summary using OpenAI
def summarize_text(input_text):
    prompt_template = PromptTemplate(template="""
    You are an intelligent assistant tasked with summarizing website content.
    Your role is to generate a concise and clear summary of the given text for contextual analysis.

    Text: {text}
    Summary:""", input_variables=["text"])

    response = llm.invoke(prompt_template.format(text=input_text))
    return response.strip()

input_text = """Feel the electrifying pulse of passion and rebellion with the ultimate rock music experience! 
    From iconic legends to rising stars, immerse yourself in the raw energy of heart-pounding guitar riffs, 
    soul-stirring vocals, and electrifying beats that ignite your spirit. 
    Whether you’re a die-hard fan or a curious newcomer, let rock music take you on a journey where every note tells a
     story and every lyric sparks emotion. Turn up the volume and live the music that defines generations – 
     because rock isn't just a genre, it's a way of life!"""
print("Generating summary using OpenAI...")
summarized_text = summarize_text(input_text)

print(f"Summarized Text: {summarized_text}")