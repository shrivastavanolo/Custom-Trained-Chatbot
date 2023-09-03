import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from streamlit.runtime.scriptrunner import get_script_run_ctx
from langchain.chains import RetrievalQA
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import pinecone
import numpy as np
from langchain.vectorstores import Pinecone

load_dotenv()

index_name = 'chattypdf'

with st.sidebar:
    st.title("ChattyPDF")
    st.markdown('''
                About:
                An LLM powered chatbot built using:
                - Streamlit
                - LangChain
                - OpenAI''')
    add_vertical_space(5)
    st.write("Made by [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)")


def main():
    
    index = pinecone.Index(index_name)

    st.header("Ask questions from your PDF files")

    uploaded_files = st.file_uploader("Upload your PDF", accept_multiple_files=True,type="pdf")
    if uploaded_files is not None:
        for pdf in uploaded_files:
            pdf_reader=PdfReader(pdf)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            m=len(chunks)
            store_name=pdf.name[:-4]
            
            embeddings=OpenAIEmbeddings()
            ember = embeddings.embed_documents(chunks)
            
            ids = map(str, np.arange(m).tolist())
            index.upsert(vectors=zip(ids,ember),namespace=store_name)
            st.write(store_name)
            
        model_name = 'text-embedding-ada-002'

        embed = OpenAIEmbeddings(
        model=model_name)

        text_field = "text"

        # switch back to normal index for langchain
        vectorstore = Pinecone(
            index, embed.embed_query,"text"
        )

        llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")
        
        ruff = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                
        tools = [
        Tool(
        name="PDF System",
        func=ruff.run,
        description="useful for when you need to answer questions about user given PDF file. Input should be a fully formed question.",
        ),]

        prefix = """Have a conversation with a human, answering the following questions as elaborately as you can. Answer in steps if possible. You have access to the following tools:"""
        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        message_history = RedisChatMessageHistory(
        url="redis://default:zyRNg3pQk44tfbpw4fQauy1lsuacbDdA@redis-19775.c10.us-east-1-4.ec2.cloud.redislabs.com:19775", ttl=600, session_id=get_script_run_ctx.session_id
        )

        memory = ConversationBufferWindowMemory(k=5,
            memory_key="chat_history", chat_memory=message_history
        )
        
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )


        query=st.text_input("Ask question about your files: ")
        
        
        if query:
            res=agent_chain.run(input=query)
            st.write(res)

    # index.delete(delete_all=True)

if __name__=='__main__':
    main()
