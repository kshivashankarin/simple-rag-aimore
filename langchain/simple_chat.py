from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

while True:
    user_msg = input("User Message : ")

    if user_msg == "exit":
        break

    response = llm.invoke(user_msg)

    print("AI response : " + response.content)





depends on user question => LLm => SQL  => generate sql query => db hit =>       => LLM => final_output
                                => RAG  => vector DB retrive relevent doc =>     => LLM => final_output


1. depends on user question => LLm => SQL/RAG 

2. generate sql query => db hit => response

3. vector DB retrive relevent doc => 3 page

4. generate final prompt (user msg : how many employees in my company? )

5.  fial_prompt => LLM => final_output


Nodes

1. decision_maker
2. generate_response_from_db
3. retrive_relevant_doc_from_rag
4. generate_final_prompt
5. generate_final_reponse





add_edge(start , decision_maker)
add_condition_edge(
        "decision_maker", {"SQL" : "sql_agent", "RAG" : "rag_agent"}
        )
add_edge("sql_agent" , "generate_final_prompt")
add_edge("rag_agent" , "generate_final_prompt")
add_edge("generate_final_prompt" , "generate_final_reponse")
add_edge("generate_final_reponse" , "END")



from langgraph.graph import StateGraph, START, END

# Create graph
graph = StateGraph(dict)

# Add nodes
graph.add_node("node1", decision_maker)
graph.add_node("node2", sql_agent)
graph.add_node("node3", rag_agent)
graph.add_node("node3", rag_agent)
graph.add_node("node4", generate_final_prompt)
graph.add_node("node5", generate_final_response)

# Add edges
graph.add_edge(START, "node1")

graph.add_conditional_edges(
    "node1",
    lambda state: state["route"],   # returns "SQL" or "RAG"
    {
        "SQL": "node2",
        "RAG": "node3",
    },
)

graph.add_edge("node2", "node4")
graph.add_edge("node3", "node4")

graph.add_edge("node4", "node5")

graph.add_edge("node5", END)

# Compile graph
app = graph.compile()


def decision_maker():
        

        return "RAG"/"SQL"