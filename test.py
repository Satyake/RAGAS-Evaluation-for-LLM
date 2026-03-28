from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os 
import asyncio
from ragas.dataset_schema import SingleTurnSample

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
#user_input -> query
#response -> response from LLM]
#reference-> Ground Truth
#retrieved context -> top k retrieved docs
async def test_context_precision():
    #create object of class of metrics
    #feed data
    #get score
    #utilize an LLM to use as an argument to scan for relevance
    llm1=ChatOpenAI(model='gpt-5.2', temperature=0)
    llm1=LangchainLLMWrapper(llm1)
    context_precision=LLMContextPrecisionWithoutReference(
        llm=llm1
    )

    #feed data-
    sample=SingleTurnSample(
        user_input="What is iffel tower?",
        response="The Eiffel Tower is in Paris and was completed in 1889.",
        retrieved_contexts=[
'Mount Everest is the tallest mountain, standing at 8,848 meters.',
 'Great Wall of China is a historic landmark spanning over 13,000 miles.',
'The Eiffel Tower is a landmark in Paris that was completed in **1889**'
        ]
    )


    #score 
    score= await context_precision.single_turn_ascore(sample)
    return print(f"Context Precision: {score}")
asyncio.run(test_context_precision())


