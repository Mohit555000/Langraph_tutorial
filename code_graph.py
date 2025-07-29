from typing_extensions import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel
load_dotenv()
client=OpenAI()

class ClassifyMessageResponse(BaseModel):
    is_coding_quesion:bool

class CodeAccuracyResponse(BaseModel):
    accuracy_percentage:str

class State(TypedDict):
    user_query:str
    llm_result:str | None
    accuracy_percentage:str | None
    is_codeing_question:bool | None

def classify_message(state:State):
    print("☢️ Inside Classify Message")
    user_query=state["user_query"]

    SYSTEM_PROMPT="""
    You are an AI Assistant. Your job is to detect if the user's query is related to coding question or not.
    Return the response in specified boolean JSON Only
    """

    response=client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=ClassifyMessageResponse,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_query}
        ]
    )
    is_coding_question=response.choices[0].message.parsed.is_coding_quesion
    state["is_codeing_question"]=is_coding_question

    return state

def route_query(state:State)->Literal["nonCoding_query","coding_query"]:
    print("☢️ Inside route Query")
    is_coding=state["is_codeing_question"]
    if is_coding:
        return "coding_query"
    return "nonCoding_query"

def nonCoding_query(state:State):
    print("☢️ Inside nonCoding_query")
    user_query=state["user_query"]
    response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"user","content":user_query}
        ]
    )
    state["llm_result"]=response.choices[0].message.content
    return state

def coding_query(state:State):
        print("☢️ inside coding_query")
        user_query=state["user_query"]
        SYSTEM_PROMPT="""
            You are a Coding Expert Agent
        """
        response=client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_query}
        ]
        )
        state["llm_result"]=response.choices[0].message.content
        return state

def coding_validate_query(state:State):
    print("☢️ inside coding_validate_query")
    user_query=state["user_query"]
    llm_result=state["llm_result"]

    SYSTEM_PROMPT=f"""
    You are expect in calculating accuracy of the code according to the question.
    User Query : {user_query}
    Code : {llm_result}
    """
    response=client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=CodeAccuracyResponse,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_query}
        ]
    )
    state["accuracy_percentage"]=response.choices[0].message.parsed.accuracy_percentage
    return state

graph_builder=StateGraph(State)

#define nodes
graph_builder.add_node("classify_message",classify_message)
graph_builder.add_node("route_query",route_query)
graph_builder.add_node("nonCoding_query",nonCoding_query)
graph_builder.add_node("coding_query",coding_query)
graph_builder.add_node("coding_validate_query",coding_validate_query)

#defining edges
graph_builder.add_edge(START,"classify_message")
graph_builder.add_conditional_edges("classify_message",route_query)
graph_builder.add_edge("nonCoding_query",END)
graph_builder.add_edge("coding_query","coding_validate_query")
graph_builder.add_edge("coding_validate_query",END)

graph=graph_builder.compile()

def main(): 
    user_query=input("> ")

    _state={
        "user_query":user_query,
        "llm_result": None,
        "accuracy_percentage": None,
        "is_codeing_question":False
    }

    response=graph.invoke(_state)
    print(response)
main()
