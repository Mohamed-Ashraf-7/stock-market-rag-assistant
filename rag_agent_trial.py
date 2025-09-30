from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import google.generativeai as genai
from keys import GOOGLE_API_KEY
import os

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
embedings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

pdf_path = "RAG_Trial_solo/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file at location:{pdf_path} not found")

# try to open and read the pdf other than this raise exception of a problem
pdf_loader = PyPDFLoader(pdf_path)
try:
    pages = pdf_loader.load()
    print(f"PDF loaded succesfully and it contains {len(pages)} pages.")
except Exception as e:
    print(f"Error loading your PDF:{str(e)}")
    raise

# split text and make page_split and the persist dir and the collection name and try to vectorstore = chroma.from docs and raise for error
# This is used to split texts and how much each chunk and the part that overlaps to keep track of the story


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
page_split = text_splitter.split_documents(pages)
database_directory = r"C:\\Courses\\LangGraph\\RAG_Trial_solo"
collection_name = "Stock_Market_Performance_2024"
try:
    vectorstore = Chroma.from_documents(  # creates a chromadb of the chunks he created
        documents=page_split,
        embedding=embedings,
        persist_directory=database_directory,
        collection_name=collection_name
    )
    print("\n================ CREATED CHROMADB VECTOR STORE SUCCESFULLY! ================  ")
except Exception as e:
    print(f"Issues creating chromaDB:{str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Number of returned chunks the default is 4
)

# create retriever tool that searches and come back with chuncks of similar meaning


@tool
def retrieve_tool(query: str) -> str:
    """Look for similar or look a likes in stock market document and retrieve this information """
    docs = retriever.invoke(query)
    if not docs:
        return (f"\nNo relative information inside the Stock market 2024\n")
    results = []
    for i, doc in enumerate(docs):
        # should be doc.page_content              ###
        results.append(f"Document Number {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)  # should return the "\n\n".join(result)   ###


tools = [retrieve_tool]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """This function Check if the last message contains any tool calls in it."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are a highly intelligent and precise AI research assistant. 
Your knowledge base is limited to the documents provided in your retriever tool. 
Do not answer from general knowledge or make up information.

Your goals:
1. Always use the retriever tool to search the document before answering. 
    - Retrieve ALL passages that are semantically similar (look-alike) to the query. 
    - If multiple relevant chunks exist, include them all.
2. If the user's query is vague, incomplete, or could mean multiple things, 
    FIRST ask a clarifying follow-up question before retrieving or answering.
3. After retrieving results, carefully read the text and answer the question 
    only with the evidence found in the document. 
    - If the retrieved chunks do not fully answer, explicitly say: 
    "The document does not provide enough information."
4. Always cite or quote the document chunks you used in your response.

Your style:
- Be clear, concise, and professional. 
- If asking a follow-up, keep it short and focused. 
- Never invent details outside of the document.

You must ground every response in the retriever tool's output.
"""
tools_dictionary = {our_tool.name: our_tool for our_tool in tools}

# HERE DFEINE YOUR LLM AGENT


def call_agent(state: AgentState) -> AgentState:
    """Function to CAll the llm to current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the llm's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for tool in tool_calls:
        print(
            f"Calling tool: {tool['name']} with query: {tool['args'].get('query', 'no query provided')}")

        if tool['name'] not in tools_dictionary:
            result = "Incorrect Tool name please retry and select one from the available tool list."
        else:
            result = tools_dictionary[tool['name']].invoke(
                tool['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(
            ToolMessage(tool_call_id=tool['id'],
                        name=tool['name'], content=str(result))
        )
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_agent)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()


def run():
    print("\n================ RAG AGENT ================\n")
    while True:
        user_input = input("\nWhat would you like to know ? ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("\n================ CHAT ENDED ================\n")
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n================ ANSWER ================\n")
        rag_responose = result['messages'][-1]
        print(rag_responose.content)


run()
