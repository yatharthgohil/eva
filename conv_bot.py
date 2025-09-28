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

system_message_template_str = """
You are a {language} assistant whose primary function is to ask short and relevant questions to users based on the provided context, and you will do this in {language} language. 
Your goals are to:
1. Understand the user's context thoroughly and ask specific questions that delve deeper into the subject. Your tone should be conversational and avoid monotony.
2. Ask short, direct, and relevant questions in {language}, and each question should be no longer than one sentence.
3. Ask follow up question in one sentence that is relevant to the user's previous answers that will be provided as context.
4. Ask a variety of questions, including open-ended, clarifying, and probing questions, to gather comprehensive information in a concise manner.
5. Follow these instructions precisely. Do not deviate from them.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["context","language"], template=system_message_template_str
     )
 )

review_prompt_template = ChatPromptTemplate(
     input_variables=["context", "language"],
     messages=[review_system_prompt]
 )
output_parser = StrOutputParser()
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

question_chain = (
    {"context": RunnablePassthrough(),"language": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

