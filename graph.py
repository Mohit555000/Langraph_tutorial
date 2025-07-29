from typing_extensions import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
load_dotenv()
client=OpenAI()
class State(TypedDict):
    query:str
    llm_result:str | None

def chat_bot(state:State):
    query=state["query"]

    llm_response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"user","content":query}
        ]
    )

    state["llm_result"]=llm_response.choices[0].message.content


    return state

graph_builder=StateGraph(State)

graph_builder.add_node("char_bot",chat_bot)

graph_builder.add_edge(START,"char_bot")
graph_builder.add_edge("char_bot",END)

graph=graph_builder.compile()


def main():
    user_query=input("> ")

    _state={
        "query":user_query,
        "llm_result":None
    }

    graph_result=graph.invoke(_state)

    print("Graph Result",graph_result)

main()
