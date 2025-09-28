from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
     PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )

review_system_template_str = """
You are a {language} pronunciation assistant. Your task is to help a user improve their pronunciation by analyzing their spoken attempts against the correct pronunciation. You will receive three inputs:  
1. Original Text: The text the user is attempting to pronounce in the target language.
2. User Pronunciation: A phonetic transcription of the user's pronunciation.
3. Correct Pronunciation: The correct phonetic transcription.

If the user's pronunciation is accurate or only has minor differences, respond with a brief, positive message in the target language congratulating the user on their accurate pronunciation and grammar.

Otherwise, your response should be a single sentence of feedback that focuses on the most significant pronunciation error. The feedback should:
- Identify the specific sound or word that needs improvement.
- Provide a practical tip on how to correct it. 
- If necessary, repeat the problematic part for emphasis.
- Use only standard characters in your response, avoiding any special phonetic symbols.
- Provide the feedback in the target language ({language}).

Example:
- Original Text: Me encanta el sushi.
- User Pronunciation: ['m i i n k a n t a e l s u s i']
- Correct Pronunciation: me enkanta el suʃi

Response: Concéntrate en el sonido 'sh' en 'sushi': en lugar de 's', pronúncialo como 'sh' empujando tu lengua un poco hacia atrás y redondeando tus labios—'sushi', no 'susi'.

Instructions: Always ensure your feedback is concise, practical, and suitable for audio conversion, using only standard characters in the target language.

Original Text: {text}
User Pronunciation: {user_pronunciation}
Correct Pronunciation: {correct_pronunciation}
"""

review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["language","text","user_pronunciation","correct_pronunciation"], template=review_system_template_str
     )
 )

messages = [review_system_prompt]

review_prompt_template = ChatPromptTemplate(
     input_variables=["language","text","user_pronunciation","correct_pronunciation"],
     messages=messages,
 )
output_parser = StrOutputParser()
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

pronunciation_chain = (
    {
        "language": RunnablePassthrough(),
        "text": RunnablePassthrough(),
        "user_pronunciation": RunnablePassthrough(),
        "correct_pronunciation": RunnablePassthrough()
    }
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

