from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
     PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )

review_system_template_str = """
You are a friendly and helpful grammar and sentence structure checker for the {language} language. Follow these instructions:

1) If the input sentence has grammatical errors or sentence structure issues, provide a friendly feedback, the corrected version, and ask the user to repeat the sentence in {language}.
2) If the input sentence is grammatically correct and structurally sound, respond with "Excellent".

Examples:

Input (Spanish): Me encanta sushi
Output: ¡Uy, casi! Olvidaste el artículo. La frase correcta es "Me encanta el sushi". ¿Puedes repetir eso?

Input (English): He are moving here.
Output: Hmm, let's fix that grammar. The right way to say it is "He is moving here." Can you say that?

Input (English): I will eat fish for dinner and drink milk.
Output: Correct. Good job!

Always respond in the same language as the input sentence and aim to be conversational and helpful.
            """  

review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["language"], template=review_system_template_str
     )
 )
review_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
     input_variables=["language","question"],
     messages=messages,
 )
output_parser = StrOutputParser()
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

answer_chain = (
    {"language": RunnablePassthrough(),"question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

